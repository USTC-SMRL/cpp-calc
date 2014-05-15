#ifndef QUANTUM_HPP
#define QUANTUM_HPP

#include <vector>
#include <complex>
#include <functional>
#include <Eigen/Eigen>
#include <ostream>

/* we often set hbar=1 */
#ifndef HBAR
#define HBAR 1
//#define HBAR 1.05457168E-34
#endif

namespace Quantum {

/* physics constants */
constexpr double hbar = (HBAR);
//-----------------------------------------------------------------------------------

/* C++ capsulation of operators in physics */

/* Operator class */
class Operator {

	bool null_identity = false;
	/* The variable named "subspace_dim" store the dimension of subspaces which this operator is in.
	 * the subspaces is numbered one by one from zero.  The value of subspace_dim[a] is the dimension
	 * of subspace numbered a.  This means that, for an operator in subspace numbered 6 and 7, subspace_dim
	 * will have 8 elements, the first 6 of which have no use.  In this case, the values of these 6
	 * elements must be set to any integer less than or equal to 0.  The reason for designing like that
	 * is for simplicity, because we won't have a large amount subspaces because of the difficulty in quantum
	 * many-body problem.  So the subspaces must be numbered one by one from zero, giving a subspace a large
	 * number won't lead to mistakes in the result, but will cause serious waste in memory and computing time.
	 */
	std::vector<int> subspace_dim;
	
	/* the variable "mat" stores the corresponding matrix of this operator.  Subspaces will be ordered by its number
	 * for example the operator B*A where B is in space 1 and A is in space 0, the matrix of B*A will be A@B
	 * where @ stands for kronecker product
	 */
	Eigen::MatrixXcd mat; 
	
public:

	/* test whether Operator is in a space*/
	bool in(int subspace)const{
		return subspace_dim[subspace]>0;
	}
	
	/* expand current operator to a larger Hilbert space
	 * the result operator will be in the direct product space of A and B
	 * where A is current operator's space and B is the space specified by parameter "subspace"
	 * the dimension of B is given by the parameter "dimension"
	 */
	Operator expand(int subspace,int dimension) const {
		std::vector<int> dim_info = subspace_dim;
		if(subspace+1>static_cast<signed int>(dim_info.size()))
			dim_info.resize(subspace+1,0);
		if(dim_info[subspace]>0)
			throw "Operator::expand(): already in subspace";
		dim_info[subspace] = dimension;
		/* here we define several terms: non-empty, lspace, rspace and espace
		 * we say a subspace numbered n is non-empty if subspace_dim[n]>0 
		 * lspace is the direct product space of non empty spaces numbered 0,1,2,...,(subspace-1)
		 * rspace is the direct product space of non empty spaces numbered (subspace+1),(subspace+2),...,n
		 * espace is the space numbered "subspace"(the parameter given)
		 */
		int new_dim;	/* dimension of the result (i.e. the direct product space of lspace, espace and rspace) */
		int ldim;		/* dimension of lspace */
		int rdim;		/* dimension of rspace */
		int rdim2;		/* dimension of the direct product space of espace and rspace */
		/* calculate new_dim, ldim, rdim and rdim2 */
		int mcol = mat.cols();
		/* if the operator before expand is null */
		if(mcol==0)
			return null_identity?Operator(subspace,Eigen::MatrixXcd::Identity(dimension,dimension)):Operator(subspace,Eigen::MatrixXcd::Zero(dimension,dimension));
		/* if the operator before expand is not null */
		new_dim = mcol*dimension;
		ldim = accumulate(dim_info.begin(),dim_info.begin()+subspace,1,
						  [](int a,int b){ return (a<=0?1:a)*(b<=0?1:b); });
		rdim2 = new_dim/ldim;
		rdim = rdim2/dimension;
		Eigen::MatrixXcd ret(new_dim,new_dim);
		/* calculate new elements */
		#pragma omp parallel for
		for(int i=0;i<new_dim;i++) {
			for(int j=0;j<new_dim;j++) {
				/* (i1,j1) is (i,j)'s coordinate in lspace */
				int i1 = i/rdim2;
				int j1 = j/rdim2;
				/* (i2,j2) is (i,j)'s coordinate in espace */
				int i2 = i%rdim2/rdim;
				int j2 = j%rdim2/rdim;
				/* (i3,j3) is (i,j)'s coordinate in rspace */
				int i3 = i%rdim2%rdim;
				int j3 = j%rdim2%rdim;
				/* (i4,j4) is (i,j)'s corresponding coordinate in the direct procuct space of lspace and rspace */
				int i4 = i1*rdim+i3;
				int j4 = j1*rdim+j3;
				ret(i,j) = (i2!=j2?0:mat(i4,j4));
			}
		}
		return Operator(dim_info,ret);
	}
	
	/* expand current operator to the product space of op(given by parameter) and this operator
	 * note that the product may not be direct product
	 */ 
	Operator expand(const Operator &op) const {
		Operator ret = *this;
		std::vector<int> target_dim = subspace_dim;
		int op_sz = op.subspace_dim.size();
		if(static_cast<signed int>(target_dim.size())<op_sz)
			target_dim.resize(op_sz,0);
		auto it1 = target_dim.begin();
		auto it2 = op.subspace_dim.begin();
		while(it2!=op.subspace_dim.end()){
			if(*it1<=0&&*it2<=0)
				goto end;
			if(*it1==*it2)
				goto end;
			if(*it1>0&&*it2>0)
				throw "Operator::expand(): dimension information mismatch";
			if(*it2>0)
				ret = ret.expand(it1-target_dim.begin(),*it2);
		end:
			++it1;
			++it2;
		}
		return ret;
	}
	
	Operator() = default;
	explicit Operator(int i) {
		if(i==1) null_identity=true;
		else if(i!=0) throw "Operator::Operator(): null operator must be zero or identity";
	}
	Operator(std::vector<int> subspace_dim,const Eigen::MatrixXcd &matrix):subspace_dim(subspace_dim),mat(matrix){
		if(subspace_dim.size()==0)
			throw "Operator::Operator(): the operator must be in at least one subspace";
		int dim = accumulate(subspace_dim.begin(),subspace_dim.end(),1,
							 [](int a,int b){ return (a<=0?1:a)*(b<=0?1:b); });
		if(dim!=matrix.cols()||dim!=matrix.rows())
			throw "Operator::Operator(): matrix size and dimension information mismatch";
	}
	/* initialize an operator in a single subspace */
	Operator(int subspace,const Eigen::MatrixXcd &matrix):subspace_dim(subspace+1,0),mat(matrix) {
		if(subspace<0)
			throw "Operator::Operator(): subspace can't be negative";
		int dim1,dim2;
		dim1 = mat.rows();
		dim2 = mat.cols();
		if(dim1!=dim2)
			throw "Operator::Operator(): matrix is not square ";
		subspace_dim[subspace] = dim1;
	}
	
	/* return the matrix of this operator */
	const Eigen::MatrixXcd &matrix() const { return mat; }
	Eigen::MatrixXcd &matrix() { return mat; }
	
	/* trace of the operator */
	std::complex<double> tr() const {
		return mat.trace();
	}
	
	/* partial trace of the operator 
	 * this function make use of C++11's feature of variadic templates.
	 * this feature makes it possible to pass arbitary number of parameters to function
	 * 
	 * to use this function, just write:
	 * operator1.tr(subspace1,subspace2,subspace3,....)
	 * 
	 * to get more information about variadic templates,
	 * see Gregoire, Solter and Kleper's book :
	 * Professional C++, Second Edition  chapter 20.6
	 */
	template <typename ... Tn>
	Operator tr(int subspace,Tn ... args) const {
		return tr(subspace).tr(args...);
	}
	Operator tr(int subspace) const {
		/* here we define several terms: non-empty, lspace, rspace and tspace
		 * we say a subspace numbered n is non-empty if subspace_dim[n]>0 
		 * lspace is the direct product space of non empty spaces numbered 0,1,2,...,(subspace-1)
		 * rspace is the direct product space of non empty spaces numbered (subspace+1),(subspace+2),...,n
		 * tspace is the Hilbert space to be traced
		 */
		int dim;		/* dimension of tspace */
		int new_dim;	/* dimension of the result (i.e. the direct product space of lspace and rspace) */
		int ldim;		/* dimension of lspace */
		int rdim;		/* dimension of rspace */
		int rdim2;		/* dimension of the direct product space of tspace and rspace */
		/* if no information stored, return *this */
		if(subspace>=static_cast<signed int>(subspace_dim.size()))
			return *this;
		dim = subspace_dim[subspace];
		if(dim<=0)
			return *this;
		/* calculate new_dim, ldim, rdim and rdim2 */
		new_dim = mat.cols()/dim;
		ldim = accumulate(subspace_dim.begin(),subspace_dim.begin()+subspace,1,
						  [](int a,int b){ return (a<=0?1:a)*(b<=0?1:b); });
		rdim = new_dim/ldim;
		rdim2 = rdim*dim;
		/* calculate the result matrix */
		Eigen::MatrixXcd ret(new_dim,new_dim);
		#pragma omp parallel for
		for(int i=0;i<new_dim;i++) {
			for(int j=0;j<new_dim;j++) {
				/* (i1,j1) is (i,j)'s coordinate in lspace */
				int i1 = i/rdim;
				int j1 = j/rdim;
				/* (i2,j2) is (i,j)'s coordinate in rspace */
				int i2 = i%rdim;
				int j2 = j%rdim;
				ret(i,j) = 0;
				for(int k=0;k<dim;k++) {/* (k,k) is the coordinate in tspace */
					int i3 = i1*rdim2+k*rdim+i2;
					int j3 = j1*rdim2+k*rdim+j2;
					ret(i,j) += mat(i3,j3);
				}
			}
		}
		/* generate the new operator */
		std::vector<int> dim_info = subspace_dim;
		dim_info[subspace] = 0;
		return Operator(dim_info,ret);
	}
	
	/* arithmetic of operator */
	Operator operator+(const Operator &rhs) const {
		Operator _lhs = expand(rhs);
		Operator _rhs = rhs.expand(*this);
		return Operator(_lhs.subspace_dim,_lhs.mat+_rhs.mat);
	}
	Operator operator-(const Operator &rhs) const {
		Operator _lhs = expand(rhs);
		Operator _rhs = rhs.expand(*this);
		return Operator(_lhs.subspace_dim,_lhs.mat-_rhs.mat);
	}
	Operator operator*(const Operator &rhs) const {
		Operator _lhs = expand(rhs);
		Operator _rhs = rhs.expand(*this);
		return Operator(_lhs.subspace_dim,_lhs.mat*_rhs.mat);
	}
	Operator operator*(const std::complex<double> &c) const {
		return Operator(subspace_dim,c*mat);
	}
	Operator operator/(const std::complex<double> &c) const {
		return Operator(subspace_dim,mat/c);
	}
	template<typename T>
	Operator &operator+=(const T &rhs) {
		return (*this = operator+(rhs));
	}
	/* a*=b is equivalent to a=a*b which may not equal to a=b*a */
	template<typename T>
	Operator &operator*=(const T &rhs) {
		return (*this = operator*(rhs));
	}
	template<typename T>
	Operator &operator-=(const T &rhs) {
		return (*this = operator-(rhs));
	}
	template<typename T>
	Operator &operator/=(const T &rhs) {
		return (*this = operator/(rhs));
	}
	Operator operator+() const {
		return *this;
	}
	Operator operator-() const {
		return Operator(subspace_dim,-mat);
	}
	
	/* Hermitian conjugate of this operator */
	Operator operator*() const {
		return Operator(subspace_dim,mat.adjoint());
	}
	
	/* calling H.U() returns a function (double->Operator): t->exp(-i*H*t/hbar)
	 * to call U(), H must be a Hermitian operator.  If this condition is violated
	 * you won't get the correct result.
	 * both this function and the function returned has a space complexity N^2
	 */
	std::function<Operator(double)> U() {
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(mat);
		auto eigenvalues  = es.eigenvalues();
		auto eigenvectors = es.eigenvectors();
		auto dim_info = subspace_dim;
		return [eigenvalues,eigenvectors,dim_info](double t) -> Operator {
			int n = eigenvalues.size();
			Eigen::MatrixXcd ret = Eigen::MatrixXcd::Zero(n,n);
			for(int i=0;i<n;i++)
				ret(i,i) = exp(std::complex<double>(0,-1)*t*eigenvalues[i]/hbar);
			ret = eigenvectors*ret*eigenvectors.adjoint();
			return Operator(dim_info,ret);
		};
	}
	
	/* A.same_space(B) tests whether A and B are in the same space */
	bool same_space(const Operator &op) const {
		auto it1 = subspace_dim.begin();
		auto it2 = op.subspace_dim.begin();
		while(it1!=subspace_dim.end()&&it2!=op.subspace_dim.end()){
			if((*it1>0||*it2>0)&&(*it1!=*it2))
				return false;
			++it1;
			++it2;
		}
		while(it1!=subspace_dim.end()){
			if(*it1>0)
				return false;
			++it1;
		}
		while(it2!=op.subspace_dim.end()){
			if(*it2>0)
				return false;
			++it2;
		}
		return true;
	}

	/* output the dimension of this operator */
	int dim() {
		return mat.cols();
	}
	
	/* return the matrix element */
	std::complex<double> &operator()(int i,int j) {
		return mat(i,j);
	}
	
	/* return the matrix element, const version */
	const std::complex<double> &operator()(int i,int j) const {
		return mat(i,j);
	}
	
	/* overload << for Operator */
	auto operator<<(std::complex<double> c) -> decltype(mat<<c) {  
		return mat<<c;
	}
};

Operator operator*(std::complex<double> c,const Operator op){
	return op*c;
}

/* instead of writing op1.tr(...) we can also write tr(op1,...) */
template <typename ... Tn>
auto tr(const Operator &op,Tn ... args) -> decltype(op.tr(args...)) {
	return op.tr(args...);
}

/* Calculate tr(AB), A and B must have the same space information.
 * This algorithm avoid calculating matrix product, which takes N^3 time.
 * Instead, this algorithm is O(N^2)
 */
std::complex<double> tr_of_prod(const Operator &A,const Operator &B) {
	if(!A.same_space(B))
		throw "tr_of_prod(): dimension information mismatch";
	const Eigen::MatrixXcd &matA = A.matrix();
	const Eigen::MatrixXcd &matB = B.matrix();
	std::complex<double> ret = 0;
	int n = matA.cols();
	for(int i=0;i<n;i++)
		for(int j=0;j<n;j++)
			ret += matA(i,j)*matB(j,i);
	return ret;
}

/* zero and identity operator */
Operator O(int subspace,int dim) {
	return Operator(subspace,Eigen::MatrixXcd::Zero(dim,dim));
}
Operator I(int subspace,int dim) {
	return Operator(subspace,Eigen::MatrixXcd::Identity(dim,dim));
}

/* helper function, won't used by user , used by function Op */
void Op_helper(Eigen::CommaInitializer<Eigen::MatrixXcd> &initializer,std::complex<double> arg1) {
	initializer,arg1;
}
template <typename ... Tn>
void Op_helper(Eigen::CommaInitializer<Eigen::MatrixXcd> &initializer,std::complex<double> arg1,Tn ... args) {
	Op_helper((initializer,arg1),args...);
}
/* this function will be used to generate an arbitary dimension operator
 * to generate a N dimension operator in subspace numbered s1 with matrix
 * element e1,e2,e3,...,eN, just write:
 * Op<N>(s1,e1,e2,....,eN);
 */
template <int n,typename ... Tn>
Operator Op(int subspace,std::complex<double> arg1,Tn ... args) {
	Eigen::MatrixXcd mat(n,n);
	Eigen::CommaInitializer<Eigen::MatrixXcd> initializer = (mat<<arg1);
	Op_helper(initializer,args...);
	return Operator(subspace,mat);
}

/* overload << for Operator */
std::ostream &operator<<(std::ostream &output,const Operator &op){  
	return output << op.matrix();
}

class Gates {
	Gates() = delete;
public:
	static Operator Hadamard(int subspace){
		return Op<2>(subspace, 1, 1,
		                       1,-1)/std::sqrt(2);
	}
	static Operator PauliX(int subspace){
		return Op<2>(subspace, 0,1,
		                       1,0);
	}
	static Operator PauliY(int subspace){
		std::complex<double> i = std::complex<double>(0,1);
		return Op<2>(subspace, 0,-i,
		                       i, 0);
	}
	static Operator PauliZ(int subspace){
		return Op<2>(subspace, 1, 0,
		                       0,-1);
	}
	static Operator PhaseShift(int subspace,double theta){
		std::complex<double> i = std::complex<double>(0,1);
		return Op<2>(subspace, 1,0,
		                       0,std::exp(i*theta));
	}
	static Operator Swap(int subspace1,int subspace2){
		Operator ret = O(subspace1,2)*O(subspace2,2);
		ret(0,0) = ret(1,2)	= ret(2,1) = ret(3,3) = 1;
		return ret;
	}
	static Operator Controlled(int subspace,const Operator &op){
		if(op.in(subspace))
			throw "Gates::Controlled(): op should no in subspace";
		Operator bit0 = Op<2>(subspace, 1,0,
		                                0,0);
		Operator bit1 = Op<2>(subspace, 0,0,
		                                0,1);
		return bit0 + bit1*op;
	}
};

}

#endif