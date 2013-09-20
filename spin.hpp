#ifndef SPIN_HPP
#define SPIN_HPP

#include <cmath>
#include <complex>
#include <Eigen/Eigen>

/* we often set hbar=1 while dealing with problems about spin */
#define HBAR 1

/* if M_PI is not defined by compiler, define it */
#ifndef M_PI
#define M_PI acos(-1)
#endif

#include "quantum.hpp"

namespace spin {
//-----------------------------------------------------------------------------------

/* mathematic constants */
constexpr double pi = M_PI;

//-----------------------------------------------------------------------------------

/* useful literal constants ( a C++11-only feature ) */

/* automatically convert these Units to SI */
double operator "" _Hz (long double f) {
	return static_cast<double>(f);
}
double operator "" _Hz (unsigned long long f) {
	return static_cast<double>(f);
}
double operator "" _KHz (long double f) {
	return 1e3*static_cast<double>(f);
}
double operator "" _KHz (unsigned long long f) {
	return 1e3*static_cast<double>(f);
}
double operator "" _kHz (long double f) {
	return 1e3*static_cast<double>(f);
}
double operator "" _kHz (unsigned long long f) {
	return 1e3*static_cast<double>(f);
}
double operator "" _MHz (long double f) {
	return 1e6*static_cast<double>(f);
}
double operator "" _MHz (unsigned long long f) {
	return 1e6*static_cast<double>(f);
}
double operator "" _GHz (long double f) {
	return 1e9*static_cast<double>(f);
}
double operator "" _GHz (unsigned long long f) {
	return 1e9*static_cast<double>(f);
}
double operator "" _THz (long double f) {
	return 1e12*static_cast<double>(f);
}
double operator "" _THz (unsigned long long f) {
	return 1e12*static_cast<double>(f);
}
double operator "" _ns (long double f) {
	return static_cast<double>(f)/1e9;
}
double operator "" _ns (unsigned long long f) {
	return static_cast<double>(f)/1e9;
}
double operator "" _us (long double f) {
	return static_cast<double>(f)/1e6;
}
double operator "" _us (unsigned long long f) {
	return static_cast<double>(f)/1e6;
}
double operator "" _ms (long double f) {
	return static_cast<double>(f)/1e3;
}
double operator "" _ms (unsigned long long f) {
	return static_cast<double>(f)/1e3;
}
double operator "" _T (long double f) {
	return static_cast<double>(f);
}
double operator "" _T (unsigned long long f) {
	return static_cast<double>(f);
}
double operator "" _G (long double f) {
	return static_cast<double>(f)/1e4;
}
double operator "" _G (unsigned long long f) {
	return static_cast<double>(f)/1e4;
}

/* an easier way to input complex number */ 
std::complex<double> operator "" _i (long double f) {
	return std::complex<double>(0,static_cast<double>(f));
}
std::complex<double> operator "" _i (unsigned long long f) {
	return std::complex<double>(0,static_cast<double>(f));
}

//-----------------------------------------------------------------------------------

/* spin operators */
/* related formula (see Zhu Dongpei's textbook of quantum mechanics):
 * <m|Jx|m'> = (hbar/2)  * { sqrt[(j+m)(j-m+1)]*delta(m,m'+1) + sqrt[(j-m)(j+m+1)]*delta(m,m'-1) }
 * <m|Jy|m'> = (hbar/2i) * { sqrt[(j+m)(j-m+1)]*delta(m,m'+1) - sqrt[(j-m)(j+m+1)]*delta(m,m'-1) }
 * where |m> is engien state of Jz i.e. Jz|m> = m|m>
 */
quantum::Operator Sx(int subspace,int dim=2) {
	Eigen::MatrixXcd mat(dim,dim);
	mat.setZero();
	double j = (dim-1.0)/2;
	for(int i=0;i<dim-1;i++)
		mat(i,i+1) = 0.5*quantum::hbar*sqrt((2*j-i)*(i+1));
	for(int i=1;i<dim;i++)
		mat(i,i-1) = 0.5*quantum::hbar*sqrt(i*(2*j-i+1));
	return quantum::Operator(subspace,mat);
}
quantum::Operator Sy(int subspace,int dim=2) {
	Eigen::MatrixXcd mat(dim,dim);
	mat.setZero();
	double j = (dim-1.0)/2;
	for(int i=0;i<dim-1;i++)
		mat(i,i+1) = -0.5_i*quantum::hbar*sqrt((2*j-i)*(i+1));
	for(int i=1;i<dim;i++)
		mat(i,i-1) = 0.5_i*quantum::hbar*sqrt(i*(2*j-i+1));
	return quantum::Operator(subspace,mat);
}
quantum::Operator Sz(int subspace,int dim=2) {
	Eigen::MatrixXcd mat(dim,dim);
	mat.setZero();
	double j = (dim-1.0)/2;
	for(int i=0;i<dim;i++)
		mat(i,i) = (j-i)*quantum::hbar;
	return quantum::Operator(subspace,mat);
}

}

#endif
