#ifndef COMPILE_TIME_POW_HPP
#define COMPILE_TIME_POW_HPP

namespace cpp_calc{

template <typename T,T base,int index>
class compile_time_pow{
public:
	static constexpr T value = base*compile_time_pow<T,base,index-1>::value;
};

template <typename T,T base>
class compile_time_pow<T,base,0>{
public:
	static constexpr T value = T(1);
};

}

#endif