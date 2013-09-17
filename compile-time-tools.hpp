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


/* count the number of parameters in variadic template */
template <typename T,typename ... Tn>
class count_variadic{
public:
	static constexpr int value = 1 + count_variadic<Tn...>::value;
};

template <typename T>
class count_variadic<T>{
public:
	static constexpr int value = 1;
};

}

#endif