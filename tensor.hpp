#ifndef TENSOR_HPP
#define TENSOR_HPP

/* tensor operations in Cartesian coordinate
 */

#include <cmath>
#include <type_traits>
#include <initializer_list>
#include <algorithm>
#include "compile-time-tools.hpp"

namespace tensor_ns{

constexpr int default_dimension = 3;

template <typename scalar,int order=2,int dimension=default_dimension>
class tensor {
public:
	/* array_type is scalar[dimension][dimension]...[dimension],
	 * which is declared using the C++'s template recursion technique */
	using array_type = typename tensor<scalar,order-1,dimension>::array_type[dimension];
private:
	using generate_a_compile_error_on_invalid_order_and_dimension = typename std::enable_if<order>=0&&dimension>=1>::type;
	/* store the components of tensor, the component with
	 * index (i1,i2,...,in) is stored in components[i1][i2]...[in] */
	array_type components;
public:
	tensor() = default;
	/* uniform initialization */
	tensor(std::initializer_list<scalar> list){
		constexpr int size = compile_time_tools::pow<dimension,order>::value;
		const int in_size = list.size();
		if(in_size>size)
			throw "size mismatch";
		std::copy(list.begin(),list.end(),((scalar*)components));
		#pragma omp parallel for
		for(int i=in_size;i<size;i++)
			((scalar*)components)[i] = scalar();
	}
	/* use operator()(i1,i2,...,in) to access components[i1][i2]...[in] */
	template <typename ... Tn>
	scalar &operator()(Tn ... indexes) {
		using generate_a_compile_error_on_invalid_number_of_parameters = typename std::enable_if<order==sizeof...(indexes)>::type;
		return compile_time_tools::vardim(components,indexes...);
	}
	template <typename ... Tn>
	scalar operator()(Tn ... indexes) const {
		using generate_a_compile_error_on_invalid_number_of_parameters = typename std::enable_if<order==sizeof...(indexes)>::type;
		return compile_time_tools::vardim(components,indexes...);
	}
	/* user operator[] to regard components as a one dimension array */
	scalar &operator[](int index){
		return ((scalar*)components)[index];
	}
	scalar operator[](int index)const{
		return ((scalar*)components)[index];
	}
};

/* zero order tensor is a scalar */
template <typename scalar,int dimension>
class tensor<scalar,0,dimension> {
public:
	using array_type = scalar;
private:
	scalar value;
public:
	tensor() = default;
	tensor(scalar value):value(value){}
	/* uniform initialization */
	tensor(std::initializer_list<scalar> list){
		constexpr int size = 1;
		const int in_size = list.size();
		if(in_size>size)
			throw "size mismatch";
		if(in_size==0)
			value = scalar();
		else
			value = *list.begin();
	}
	operator scalar() const { return value; }
	tensor<scalar,0,dimension> operator=(const scalar &rhs){ value = rhs; }
	scalar &operator()() { return value; }
	scalar operator()() const { return value; }
	scalar &operator[](int index) { return *(&value+index); }
	scalar operator[](int index) const { return *(&value+index); }
};

/* one order tensor is a vector */
template <typename scalar,int dimension>
class tensor<scalar,1,dimension>{
public:
	using array_type = scalar[dimension];
private:
	array_type components;
public:
	tensor() = default;
	/* uniform initialization */
	tensor(std::initializer_list<scalar> list){
		constexpr int size = dimension;
		const int in_size = list.size();
		if(in_size>size)
			throw "size mismatch";
		std::copy(list.begin(),list.end(),components);
		#pragma omp parallel for
		for(int i=in_size;i<size;i++)
			components[i] = scalar();
	}
	scalar &operator()(int index) { return components[index]; }
	scalar operator()(int index) const { return components[index]; }
	scalar &operator[](int index){ return components[index]; }
	scalar operator[](int index) const { return components[index]; }
};
template <typename scalar,int dimension=default_dimension>
using vector = tensor<scalar,1,dimension>;

/* operators */
template <typename scalar1,typename scalar2,int order,int dimension>
tensor<decltype(scalar1()+scalar2()),order,dimension> operator+(const tensor<scalar1,order,dimension> &lhs,const tensor<scalar2,order,dimension> &rhs) {
	constexpr int size = compile_time_tools::pow<dimension,order>::value;
	tensor<decltype(scalar1()+scalar2()),order,dimension> ret;
	#pragma omp parallel for
	for(int i=0;i<size;i++)
		ret[i] = lhs[i]+rhs[i];
	return ret;
}
template <typename scalar1,typename scalar2,int order,int dimension>
tensor<decltype(scalar1()-scalar2()),order,dimension> operator-(const tensor<scalar1,order,dimension> &lhs,const tensor<scalar2,order,dimension> &rhs) {
	constexpr int size = compile_time_tools::pow<dimension,order>::value;
	tensor<decltype(scalar1()-scalar2()),order,dimension> ret;
	#pragma omp parallel for
	for(int i=0;i<size;i++)
		ret[i] = lhs[i]-rhs[i];
	return ret;
}
template <typename scalar1,typename scalar2,int order,int dimension>
tensor<decltype(scalar1()*scalar2()),order,dimension> operator*(scalar1 lhs,const tensor<scalar2,order,dimension> &rhs) {
	constexpr int size = compile_time_tools::pow<dimension,order>::value;
	tensor<decltype(scalar1()*scalar2()),order,dimension> ret;
	#pragma omp parallel for
	for(int i=0;i<size;i++)
		ret[i] = lhs*rhs[i];
	return ret;
}
template <typename scalar1,typename scalar2,int order,int dimension>
tensor<decltype(scalar1()*scalar2()),order,dimension> operator*(const tensor<scalar1,order,dimension> &lhs,scalar2 rhs) {
	constexpr int size = compile_time_tools::pow<dimension,order>::value;
	tensor<decltype(scalar1()*scalar2()),order,dimension> ret;
	#pragma omp parallel for
	for(int i=0;i<size;i++)
		ret[i] = lhs[i]*rhs;
	return ret;
}
template <typename scalar1,typename scalar2,int order,int dimension>
tensor<decltype(scalar1()/scalar2()),order,dimension> operator/(const tensor<scalar1,order,dimension> &lhs,scalar2 rhs) {
	constexpr int size = compile_time_tools::pow<dimension,order>::value;
	tensor<decltype(scalar1()/scalar2()),order,dimension> ret;
	#pragma omp parallel for
	for(int i=0;i<size;i++)
		ret[i] = lhs[i]/rhs;
	return ret;
}

/* prod of tensors */
template<typename scalar1,int order1,int dimension,typename scalar2,int order2>
tensor<decltype(scalar1()*scalar2()),order1+order2,dimension>  prod(const tensor<scalar1,order1,dimension> &lhs,const tensor<scalar2,order2,dimension> &rhs){
	tensor<decltype(scalar1()*scalar2()),order1+order2,dimension> ret;
	int rsize = compile_time_tools::pow<dimension,order2>::value;
	int size  = compile_time_tools::pow<dimension,order1+order2>::value;
	#pragma omp parallel for
	for(int i=0;i<size;i++){
		int lidx = i/rsize;
		int ridx = i%rsize;
		ret[i] = lhs[lidx]*rhs[ridx];
	}
	return ret;
}

/* contraction of tensor */
template <int index1,int index2,typename scalar,int order,int dimension>
tensor<scalar,order-2,dimension> contract(const tensor<scalar,order,dimension> &T) {
	/* check if index1 and index2 are valid, if not, generate a compile time error */
	constexpr bool valid = index1>=0 && index2>=0 && index1<order && index2<order && index1!=index2;
	using generate_a_compile_error_on_invalid_index1_and_index2 = typename std::enable_if<valid>::type;
	/* calculate */
	constexpr int lower_idx  = index1<index2?index1:index2;
	constexpr int higher_idx = index1<index2?index2:index1;
	constexpr int size  = compile_time_tools::pow<dimension,order-2>::value;
	constexpr int lsize = compile_time_tools::pow<dimension,order-higher_idx-1>::value;
	constexpr int hsize = compile_time_tools::pow<dimension,order-lower_idx-2>::value;
	tensor<scalar,order-2,dimension> ret;
	#pragma omp parallel for
	for(int i=0;i<size;i++) {
		int hidx = i/hsize;
		int midx = i%hsize/lsize;
		int lidx = i%lsize;
		ret[i] = 0;
		for(int j=0;j<dimension;j++){
			int oldidx = lidx;
			oldidx += hidx*compile_time_tools::pow<dimension,order-lower_idx>::value;
			oldidx += j*compile_time_tools::pow<dimension,order-lower_idx-1>::value;
			oldidx += midx*compile_time_tools::pow<dimension,order-higher_idx>::value;
			oldidx += j*compile_time_tools::pow<dimension,order-higher_idx-1>::value;
			ret[i] += T[oldidx];
		}
	}
	return ret;
}

/* dot product of tensor */
template<typename scalar1,int order1,int dimension,typename scalar2,int order2>
tensor<decltype(scalar1()*scalar2()),order1+order2-2,dimension> dot(const tensor<scalar1,order1,dimension> &lhs,const tensor<scalar2,order2,dimension> &rhs){
	return contract<order1-1,order1>(prod(lhs,rhs));
}

/* cross product of tensor */

}

#endif