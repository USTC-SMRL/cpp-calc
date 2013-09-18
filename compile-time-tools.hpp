#ifndef COMPILE_TIME_TOOLS_HPP
#define COMPILE_TIME_TOOLS_HPP

namespace compile_time_tools{

/* compile time integer power */
template <int base,int index>
class pow{
public:
	static constexpr int value = base*pow<base,index-1>::value;
};

template <int base>
class pow<base,0>{
public:
	static constexpr int value = 1;
};

/* Use vardim(array,i1,i2,...,in) to access array[i1][i2]...[in].
 * The returned value is a reference.
 */
template <typename array_type,typename T,typename ... Tn>
class vardim_class{
	static array_type not_used;
public:
	using scalar_type = typename vardim_class<decltype(not_used[0]),Tn...>::scalar_type;
};

template <typename array_type,typename T>
class vardim_class<array_type,T>{
	static array_type not_used;
public:
	using scalar_type = decltype(not_used[0]);
};

template <typename scalar_type,typename array_type,typename T>
scalar_type &vardim_helper(array_type array,T lindex) {
	return array[lindex];
}

template <typename scalar_type,typename array_type,typename T,typename ... Tn>
scalar_type &vardim_helper(array_type array,T lindex,Tn ... indexes) {
	return vardim_helper<scalar_type>(array[lindex],indexes...);
}

template <typename array_type,typename ... Tn>
typename vardim_class<array_type,Tn...>::scalar_type &vardim(array_type array,Tn ... indexes) {
	using scalar_type = typename vardim_class<array_type,Tn...>::scalar_type;
	return vardim_helper<scalar_type>(array,indexes...);
}

/**
 * this util uses the initial N elements in an array or object with operator[] overloaded
 * as the initial N parameters to call a function or object with operator() overloaded.
 * 
 * usage:
 * array_args<N>(func,array,other_parameters...)
 * where
 * N stands for the number of elements in array to be used as parameters
 * func stands for the function you want to call
 * array stands for the array you want to use
 * other_parameters... stands for the residual parameters of func
 * 
 * for example:
 * if there is a function and an array:
 *  void fun1(double a,double b,double c,int d);
 *  double arr[3];
 * and you want to call the function like this:
 *  fun1(arr[0],arr[1],arr[2],25);
 * you can simplify write:
 *  array_args<3>(fun1,arr,25);
 */

template <int nparams,typename function_t,typename array_t,typename ... params_t>
class array_argser {
public:
	static inline auto call(function_t &f,array_t &array,params_t...args)
		-> decltype(array_argser<nparams-1,function_t,array_t,decltype(array[nparams-1]),params_t...>::call(f,array,array[nparams-1],args...))
	{
		return array_argser<nparams-1,function_t,array_t,decltype(array[nparams-1]),params_t...>::call(f,array,array[nparams-1],args...);
	}
};

template <typename function_t,typename array_t,typename ... params_t>
class array_argser<0,function_t,array_t,params_t...> {
public:
	static inline auto call(function_t &f,array_t &array,params_t...args)->decltype(f(args...))
	{
		return f(args...);
	}
};

template <int nparams,typename function_t,typename array_t,typename...params_t>
inline auto array_args(function_t &func,array_t &array,params_t...args)
	->decltype(array_argser<nparams,function_t,array_t,params_t...>::call(func,array,args...))
{
	return array_argser<nparams,function_t,array_t,params_t...>::call(func,array,args...);
}

}

#endif