#ifndef SPIN_HPP
#define SPIN_HPP

#include <cmath>
#include <complex>
#include <Eigen/Eigen>

/* if M_PI is not defined by compiler, define it */
#ifndef M_PI
#define M_PI acos(-1)
#endif

#include <quantum.hpp>
#include <tensor.hpp>

namespace Spin {

using namespace Tensor;
using namespace Quantum;

//-----------------------------------------------------------------------------------

/* mathematic constants */
constexpr double pi = M_PI;

//-----------------------------------------------------------------------------------

/* useful literal constants ( a C++11-only feature ) */

/* useful literal constants ( a C++11-only feature ) */

/* automatically convert these Units to SI */
/* frequency */
constexpr double operator"" _Hz (long double f) {
	return static_cast<double>(f);
}
constexpr double operator"" _Hz (unsigned long long f) {
	return static_cast<double>(f);
}
constexpr double operator"" _KHz (long double f) {
	return 1e3*static_cast<double>(f);
}
constexpr double operator"" _KHz (unsigned long long f) {
	return 1e3*static_cast<double>(f);
}
constexpr double operator"" _kHz (long double f) {
	return 1e3*static_cast<double>(f);
}
constexpr double operator"" _kHz (unsigned long long f) {
	return 1e3*static_cast<double>(f);
}
constexpr double operator"" _MHz (long double f) {
	return 1e6*static_cast<double>(f);
}
constexpr double operator"" _MHz (unsigned long long f) {
	return 1e6*static_cast<double>(f);
}
constexpr double operator"" _GHz (long double f) {
	return 1e9*static_cast<double>(f);
}
constexpr double operator"" _GHz (unsigned long long f) {
	return 1e9*static_cast<double>(f);
}
constexpr double operator"" _THz (long double f) {
	return 1e12*static_cast<double>(f);
}
constexpr double operator"" _THz (unsigned long long f) {
	return 1e12*static_cast<double>(f);
}
/* time */
constexpr double operator"" _ns (long double f) {
	return static_cast<double>(f)/1e9;
}
constexpr double operator"" _ns (unsigned long long f) {
	return static_cast<double>(f)/1e9;
}
constexpr double operator"" _us (long double f) {
	return static_cast<double>(f)/1e6;
}
constexpr double operator"" _us (unsigned long long f) {
	return static_cast<double>(f)/1e6;
}
constexpr double operator"" _ms (long double f) {
	return static_cast<double>(f)/1e3;
}
constexpr double operator"" _ms (unsigned long long f) {
	return static_cast<double>(f)/1e3;
}
/* magnetic field */
constexpr double operator"" _T (long double f) {
	return static_cast<double>(f);
}
constexpr double operator"" _T (unsigned long long f) {
	return static_cast<double>(f);
}
constexpr double operator"" _G (long double f) {
	return static_cast<double>(f)/1e4;
}
constexpr double operator"" _G (unsigned long long f) {
	return static_cast<double>(f)/1e4;
}
/* length */
constexpr double operator"" _nm (long double f) {
	return static_cast<double>(f)/1e9;
}
constexpr double operator"" _nm (unsigned long long f) {
	return static_cast<double>(f)/1e9;
}
constexpr double operator"" _um (long double f) {
	return static_cast<double>(f)/1e6;
}
constexpr double operator"" _um (unsigned long long f) {
	return static_cast<double>(f)/1e6;
}
constexpr double operator"" _mm (long double f) {
	return static_cast<double>(f)/1e3;
}
constexpr double operator"" _mm (unsigned long long f) {
	return static_cast<double>(f)/1e3;
}
/* angle */
constexpr double operator"" _deg (long double f) {
	return static_cast<double>(f)*pi/180;
}
constexpr double operator"" _deg (unsigned long long f) {
	return static_cast<double>(f)*pi/180;
}

/* an easier way to input complex number */ 
constexpr std::complex<double> operator "" _i (long double f) {
	return std::complex<double>(0,static_cast<double>(f));
}
constexpr std::complex<double> operator "" _i (unsigned long long f) {
	return std::complex<double>(0,static_cast<double>(f));
}

//-----------------------------------------------------------------------------------

/* spin operators */
/* related formula (see Zhu Dongpei's textbook of quantum mechanics):
 * <m|Jx|m'> = (hbar/2)  * { sqrt[(j+m)(j-m+1)]*delta(m,m'+1) + sqrt[(j-m)(j+m+1)]*delta(m,m'-1) }
 * <m|Jy|m'> = (hbar/2i) * { sqrt[(j+m)(j-m+1)]*delta(m,m'+1) - sqrt[(j-m)(j+m+1)]*delta(m,m'-1) }
 * where |m> is engien state of Jz i.e. Jz|m> = m|m>
 */
Quantum::Operator Sx(int subspace,int dim=2) {
	Eigen::MatrixXcd mat(dim,dim);
	mat.setZero();
	double j = (dim-1.0)/2;
	for(int i=0;i<dim-1;i++)
		mat(i,i+1) = 0.5*Quantum::hbar*sqrt((2*j-i)*(i+1));
	for(int i=1;i<dim;i++)
		mat(i,i-1) = 0.5*Quantum::hbar*sqrt(i*(2*j-i+1));
	return Quantum::Operator(subspace,mat);
}
Quantum::Operator Sy(int subspace,int dim=2) {
	Eigen::MatrixXcd mat(dim,dim);
	mat.setZero();
	double j = (dim-1.0)/2;
	for(int i=0;i<dim-1;i++)
		mat(i,i+1) = -0.5_i*Quantum::hbar*sqrt((2*j-i)*(i+1));
	for(int i=1;i<dim;i++)
		mat(i,i-1) = 0.5_i*Quantum::hbar*sqrt(i*(2*j-i+1));
	return Quantum::Operator(subspace,mat);
}
Quantum::Operator Sz(int subspace,int dim=2) {
	Eigen::MatrixXcd mat(dim,dim);
	mat.setZero();
	double j = (dim-1.0)/2;
	for(int i=0;i<dim;i++)
		mat(i,i) = (j-i)*Quantum::hbar;
	return Quantum::Operator(subspace,mat);
}
Tensor::vector<Quantum::Operator,3> S(int subspace,int dim=2) {
	return { Sx(subspace,dim),Sy(subspace,dim),Sz(subspace,dim) };
}

//set default dimension of I and O to 2
Quantum::Operator O(int subspace) {
	return Quantum::O(subspace,2);
}
Quantum::Operator I(int subspace) {
	return Quantum::I(subspace,2);
}

}

#endif
