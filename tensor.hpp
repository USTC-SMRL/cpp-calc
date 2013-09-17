#ifndef TENSOR_HPP
#define TENSOR_HPP

/* tensor operations in Cartesian coordinate
 */

#include <cmath>
#include <type_traits>
#include "compile-time-tools.hpp"

namespace cpp_calc{

constexpr int default_dimension = 3;

template <typename scalar,int order=2,int dimension=default_dimension>
class tensor {
public:
	/* array_type is scalar[dimension][dimension]...[dimension],
	 * which is declared using the C++'s template recursion technique */
	using array_type = typename tensor<scalar,order-1,dimension>::array_type[dimension];
private:
	/* store the components of tensor, the component with
	 * index (i1,i2,...,in) is stored in components[i1][i2]...[in] */
	array_type components;
	/* helper function to check whether the number of parameters equate order */
	template <typename enable,typename ... Tn>
	scalar &check_order_and_get(Tn ... args) {
		return compile_time_tools::vardim(components,args...);
	}
public:
	/* use operator(i1,i2,...,in) to access components[i1][i2]...[in] */
	template <typename ... Tn>
	scalar &operator()(Tn ... indexes) {
		return check_order_and_get<typename std::enable_if<order==sizeof...(indexes)>::type>(indexes...);
	}
};

/* zero order tensor is a scalar */
template <typename scalar,int dimension>
class tensor<scalar,0,dimension> {
	scalar value;
public:
	tensor(scalar value):value(value){}
	operator scalar() const { return value; }
	tensor<scalar,0,dimension> operator=(const scalar &rhs){ value = rhs; }
};

/* one order tensor is a vector */
template <typename scalar,int dimension>
class tensor<scalar,1,dimension>{
public:
	using array_type = scalar[dimension];
private:
	array_type scalars;
public:
	scalar &operator()(int index) { return scalars[index]; }
};
template <typename scalar,int dimension=default_dimension>
using vector = tensor<scalar,1,dimension>;

/* prod of tensors */
template<typename scalar1,int order1,int dimension,typename scalar2,int order2>
tensor<decltype(scalar1()*scalar2()),order1+order2,dimension>  prod(tensor<scalar1,order1,dimension> lhs,tensor<scalar2,order2,dimension> rhs){
	tensor<decltype(scalar1()*scalar2()),order1+order2,dimension> ret;
	return;
}

/* contraction of tensor */

/* dot product of tensor */

/* cross product of tensor */

}

#endif