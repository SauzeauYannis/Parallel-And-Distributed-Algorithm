#pragma once
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cstring>

// for associativity and not commutativity test
struct iMatrix22 
{
    int values[4]; // row by row
    __host__ __device__ 
    iMatrix22(int a=0, int b=0, int c=0, int d=0) 
    {
        values[0] = a;
        values[1] = b;
        values[2] = c;
        values[3] = d;
    }
    iMatrix22(const iMatrix22&) = default;
    ~iMatrix22() = default;
    iMatrix22& operator=(const iMatrix22&) = default;      
    __host__ __device__  
    int get(const int row, const int col) const 
    { 
        return values[row*2+col]; 
    }
    __host__ __device__
    int& get(const int row, const int col)
    { 
        return values[row*2+col]; 
    }
    __host__ __device__
    iMatrix22 operator*(const iMatrix22& that) const
    {
        iMatrix22 result;
        for(int row=0; row<2; row++)
            for(int col=0; col<2; col++)
                for(int k=0; k<2; k++)
                    result.get(row,col) += get(row,k) * that.get(k,col);
        return result;
    }
    __host__ __device__
    void fill(const int value) {
        for(int i=0; i<4; ++i) values[i] = value;    
    }
    __host__ __device__
    bool operator==(const iMatrix22& that) 
    {
        for(int i=0; i<4; ++i)
            if( values[i] != that.values[i] ) 
                return false;
        return true;
    }
    __host__ __device__
    bool operator!=(const iMatrix22& that) 
    {
        for(int i=0; i<4; ++i)
            if( values[i] != that.values[i] ) 
                return true;
        return false;
    }
    static iMatrix22 make_identity() { 
        iMatrix22 result;
        result.values[0] = result.values[3] = 1;
        return result;
    }
    friend std::ostream& operator<<(std::ostream&, iMatrix22&);
};

