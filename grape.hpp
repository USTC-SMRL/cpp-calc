#ifndef GRAPE_HPP
#define GRAPE_HPP

#include <complex>
#include <functional>
#include <cstdlib>
#include <algorithm>
#include <tensor.hpp>
#include <quantum.hpp>
#include <cmath>

#ifdef DEBUG
#include <iostream>
#endif

namespace Quantum {

/* origvec is original control vector, say X[N_orig].
 * Original control vector is transformed into transvec, which is called transformed control vector, say Y[N_trans].
 * Transformed control vector is the coefficient in front of each control Hamiltonian, i.e. H = H0 + sum(k=0 to N_trans-1,Y[k]*H_ctrl[k])
 */
template <int N_orig,int N_trans,int N_sample=1>
class System{
public:

	using origvec  = Eigen::Matrix<double,1,N_orig>;
	using transvec = Eigen::Matrix<double,1,N_trans>;
	using jacobmat = Eigen::Matrix<double,N_trans,N_orig>;
	
	Operator H_natural[N_sample];
	Operator H_ctrl[N_trans];
	
	
	std::function<std::vector<origvec>(const std::vector<origvec>&)> penalty;
	std::function<void(std::vector<origvec>&)> constrain;
	std::function<transvec(origvec)> transform;
	std::function<jacobmat(origvec)> Jacobian;
	
	//useful function
	static std::vector<origvec> id_penalty(const std::vector<origvec>&v,std::function<origvec(origvec)> c){
		std::vector<origvec> ret(v.size());
		std::transform(v.begin(),v.end(),ret.begin(),c);
		return ret;
	}
	static std::vector<origvec> no_penalty(const std::vector<origvec>&v){
		return id_penalty(v,[](origvec)->origvec{ return origvec::Zero(); });
	}
	
	static origvec id_transform(origvec v){ return v; }
	
	static jacobmat id_Jacobian(origvec) { return jacobmat::Identity(); }
	
	static void no_constrain(std::vector<origvec>&){ return; }
	static void id_constrain(std::vector<origvec>& vec,std::function<origvec(origvec)> c){
		std::transform(vec.begin(),vec.end(),vec.begin(),c);
	}
	static void id_bound_constrain(std::vector<origvec>& vec,double bound){
		id_constrain(vec,[bound](origvec v)->origvec{
			for(int i=0;i<N_orig;i++){
				if(v(i)> bound) v(i) =  bound;
				if(v(i)<-bound) v(i) = -bound;
			}
			return v;
		});
	}
};

template <int N_orig,int N_trans,int N_sample>
void iter(Operator &Utarget,const System<N_orig,N_trans,N_sample> &system,
          double *_fidelity,double *_log_delta_fidelity,double *_delta_ldfidelity,double step,double tau,double threshold,
		  Operator Ueff[N_sample],std::vector<typename System<N_orig,N_trans,N_sample>::origvec> *_ctrlvec){
	using system_t = System<N_orig,N_trans,N_sample>;
	using origvec = typename system_t::origvec;
	using transvec = typename system_t::transvec;
	using jacobmat = typename system_t::jacobmat;
	
	double &fidelity = *_fidelity;
	double &log_delta_fidelity = *_log_delta_fidelity;
	double &delta_ldfidelity = *_delta_ldfidelity;
	std::vector<origvec> &ctrlvec = *_ctrlvec;
	
	const int N_time = ctrlvec.size();
	
	Operator U[N_sample][N_time];
	Operator X[N_sample][N_time];
	Operator P[N_sample][N_time];
	transvec transformed[N_time];
	jacobmat Jacobian[N_time];
	std::vector<origvec> new_ctrlvec(N_time);
	std::vector<origvec> penalty;
	
	double prev_ldfid = log_delta_fidelity;
	
	//calculate transformed control vector, penalty and Jacobian
	std::transform(ctrlvec.begin(),ctrlvec.end(),transformed,system.transform);
	std::transform(ctrlvec.begin(),ctrlvec.end(),Jacobian,system.Jacobian);
	penalty = system.penalty(ctrlvec);

	//calculate Uj
	#pragma omp parallel for
	for(int j=0;j<N_time;j++){
		Operator H[N_sample];
		for(int i=0;i<N_sample;i++){
			//Calculate Hj
			H[i] = system.H_natural[i];
			for(int k=0;k<N_trans;k++)
				H[i] += transformed[j](k)*system.H_ctrl[k];
			U[i][j] = (H[i].U())(tau);
		}
	}
	
	//calculate X[j] = U[j]...U[0]
	#pragma omp parallel for
	for(int i=0;i<N_sample;i++){
		X[i][0] = U[i][0];
		for(int j=1;j<N_time;j++) 
			X[i][j] = U[i][j]*X[i][j-1];
	}
	
	//calculate Ueff = U[N-1]U[N-1]...U[0]
	for(int i=0;i<N_sample;i++)
		Ueff[i] = X[i][N_time-1];
	
	//calculate A=(Ueff^+)*Utarget
	Operator A[N_sample];
	std::complex<double> trA[N_sample];
	for(int i=0;i<N_sample;i++){
		A[i] = (*Ueff[i])*Utarget;
		trA[i] = tr(A[i]); 
	}
	
	//calculate P[j] = (U[j+1]^+)...(U[N-1]^+)*Utarget
	#pragma omp parallel for
	for(int i=0;i<N_sample;i++){
		P[i][N_time-1] = Utarget;
		for(int j=N_time-2;j>=0;j--)
			P[i][j] = (*U[i][j+1])*P[i][j+1];
	}
	
	//update each original control vector
	#pragma omp parallel for
	for(int j=0;j<N_time;j++){
		transvec partiald_trans[N_sample];
		origvec partiald_orig[N_sample];
		Operator XPj[N_sample];
		for(int i=0;i<N_sample;i++){
			//calculate partial derivative with transformed vector
			XPj[i] = X[i][j]*(*P[i][j]);
			for(int k=0;k<N_trans;k++)
				partiald_trans[i](k) = 2*tau*std::imag(tr_of_prod(XPj[i],system.H_ctrl[k].expand(XPj[i]))*trA[i]);
		
			//calculate partial derivative with original vector
			partiald_orig[i] = partiald_trans[i]*Jacobian[j];
		}
		
		//calculate proposal change of each original vector
		origvec partiald_orig_avg = origvec::Zero();
		for(int i=0;i<N_sample;i++)
			partiald_orig_avg += partiald_orig[i];
		partiald_orig_avg /= N_sample;
		origvec dvec = step*(partiald_orig_avg+penalty[j]);
		
		//calculate expected new control vector
		new_ctrlvec[j] = ctrlvec[j] + dvec;
	}
	
	system.constrain(new_ctrlvec);
	ctrlvec = new_ctrlvec;
	
	//calculate fidelity
	double sqrfid[N_sample];
	fidelity = 0;
	for(int i=0;i<N_sample;i++){
		sqrfid[i] = abs( trA[i] / ((double)Ueff[0].dim()) );
		fidelity += sqrfid[i]*sqrfid[i];
	}
	fidelity /= N_sample;
	log_delta_fidelity = std::log(threshold-fidelity);
	delta_ldfidelity = log_delta_fidelity-prev_ldfid;
}

template <int N_orig,int N_trans,int N_sample>
double GRAPE(Operator &Utarget,System<N_orig,N_trans,N_sample> system,
             double T,double step,double threshold,double die_diff_ratio,int max_times,
			 std::vector<typename System<N_orig,N_trans,N_sample>::origvec> &ctrlvec) {
	constexpr int N_skip_first = 2;
	const int N_time = ctrlvec.size();
	
	double fidelity = 0;
	double log_delta_fidelity = 0;
	double delta_ldfidelity;
	double init_delta_ldfidelity;
	Operator Ueff[N_sample];
	
	const double tau = T/N_time;

	auto do_iter = std::bind(iter<N_orig,N_trans,N_sample>,Utarget,system,
	                         &fidelity,&log_delta_fidelity,&delta_ldfidelity,step,tau,threshold,
							 Ueff,&ctrlvec);
	
	for(int i=0;i<N_skip_first;i++)
		do_iter();
	init_delta_ldfidelity = delta_ldfidelity;
	int count = 0;
	double ratio;
	do{
		count++;
		do_iter();
		ratio = std::abs(delta_ldfidelity/init_delta_ldfidelity);
		#ifdef DEBUG
		std::cout << count << "\t" << fidelity << std::endl;
		#endif
	}while(fidelity<threshold && (ratio>die_diff_ratio) && (count<max_times||max_times<=0));
	
	//calculate the propagator gotten
	Utarget = Operator(0);
	for(int i=0;i<N_sample;i++)
		Utarget += Ueff[i];
	Utarget /= N_sample;
	
	return fidelity;
}

}

#endif