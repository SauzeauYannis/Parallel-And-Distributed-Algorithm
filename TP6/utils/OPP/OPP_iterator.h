#pragma once
#include <iterator>
#include <functional>

namespace OPP
{
        // counting iterator ... T should be an integer (char/short/int/long/long long, signed or unsigned)
    template<
        typename T=long,
        typename Tdiff=long long>
    class CountingIterator : public std::iterator<std::random_access_iterator_tag, T>
    {
        T position;
    public:
        using pointer = typename std::iterator<std::random_access_iterator_tag,T>::pointer;
        using reference = typename std::iterator<std::random_access_iterator_tag,T>::reference;
        
        CountingIterator(const T position=T(0)) 
            : position(position) 
        {}
        CountingIterator(const CountingIterator& cit) = default;
        ~CountingIterator() = default;
        CountingIterator& operator=(const CountingIterator&) = default;
        CountingIterator& operator++() { 
            ++position;
            return *this;
        }
        CountingIterator operator++(int) const {
            return CountingIterator(position++);
        }
        CountingIterator& operator--() { 
            --position;
            return *this;
        }
        CountingIterator operator--(int) const {
            return CountingIterator(position--);
        }
        bool operator==(const CountingIterator& cit) const {
            return position == cit.position;
        }
        bool operator!=(const CountingIterator& cit) const {
            return position != cit.position ;
        }
        T operator*() const { 
            return position; 
        }
        T& operator*() { 
            return position; 
        }
        CountingIterator operator+(const Tdiff& dt) const {
            return CountingIterator(T(position+dt));
        }
        CountingIterator& operator+=(const Tdiff& dt) {
            position += dt;
            return *this;
        }
        CountingIterator operator-(const Tdiff& dt) const {
            return CountingIterator(T(position-dt));
        }
        CountingIterator& operator-=(const Tdiff& dt) {
            position -= dt;
            return *this;
        }        
        T operator[](const Tdiff& n) const {
            return position+n;
        }
        bool operator<(const CountingIterator& cit) const {
            return position < cit.position;
        }

        bool operator>(const CountingIterator& cit) const {
            return position > cit.position;
        }

        bool operator<=(const CountingIterator& cit) const {
            return position <= cit.position;
        }

        bool operator>=(const CountingIterator& cit) const {
            return position >= cit.position;
        }

        Tdiff operator+(const CountingIterator& cit) const {
            return position + cit.position;
        }

        Tdiff operator-(const CountingIterator& cit) const {
            return position - cit.position;
        }
    };

    template<typename T>
    CountingIterator<T> make_counting_iterator(T position) {
        return CountingIterator<T>(position);
    }

    template<
        typename Iterator,
        typename Functor,
        typename Tsrc, 
        typename Tdst,
        typename Tdiff=long long
    >
    class TransformIterator : public std::iterator<std::random_access_iterator_tag, Tdst, Tdiff,void,void>
    {
        Functor  transform;
        Iterator iterator;

    public:
        TransformIterator(const Iterator& iterator, const Functor& transform) 
            : transform(transform), iterator(iterator) 
        {}
        TransformIterator(Iterator&& iterator, Functor&& transform) 
            : transform(transform), iterator(iterator) 
        {}
        TransformIterator(const TransformIterator&) = default;
        ~TransformIterator() = default;
        TransformIterator& operator=(const TransformIterator&) = default;
                
        TransformIterator& operator++() { 
            ++iterator;
            return *this;
        }
        TransformIterator operator++(int) const {
            auto copy = TransformIterator(iterator, transform);
            ++iterator;
            return copy;
        }
        TransformIterator& operator--() { 
            --iterator;
            return *this;
        }
        TransformIterator operator--(int) const {
            auto copy = TransformIterator(iterator, transform);
            --iterator;
            return copy;
        }
        bool operator==(const TransformIterator& cit) const {
            return iterator == cit.iterator && transform == cit.transform;
        }
        bool operator!=(const TransformIterator& cit) const {
            return ! ( *this == cit );
        }
        Tdst operator*() const { 
            return std::invoke(transform, *iterator); 
        }
        TransformIterator operator+(const Tdiff& dt) const {
            return TransformIterator(iterator+dt, transform);
        }
        TransformIterator& operator+=(const Tdiff& dt) {
            iterator += dt;
            return *this;
        }
        TransformIterator operator-(const Tdiff& dt) const {
            return TransformIterator(iterator-dt, transform);
        }
        TransformIterator& operator-=(const Tdiff& dt) {
            iterator -= dt;
            return *this;
        }        
        Tdst operator[](const Tdiff& n) const {
            return std::invoke(transform, iterator[n]);
        }
        bool operator<(const TransformIterator& cit) const {
            return iterator < cit.iterator;
        }

        bool operator>(const TransformIterator& cit) const {
            return iterator > cit.iterator;
        }

        bool operator<=(const TransformIterator& cit) const {
            return iterator <= cit.iterator;
        }

        bool operator>=(const TransformIterator& cit) const {
            return iterator >= cit.iterator;
        }

        Tdiff operator+(const TransformIterator& cit) const {
            return iterator + cit.iterator;
        }

        Tdiff operator-(const TransformIterator& cit) const {
            return iterator - cit.iterator;
        }
    };
    
    template<
        typename Iterator,
        typename Tsrc,
        typename Tdst,
        typename Tdiff=long long>
    inline
    auto make_transform_iterator(Iterator iterator, std::function<Tdst(Tsrc)> functor) 
    {
        //using Tsrc = typename Iterator::value_type;
        //using Tdst = typename functor::result_type; // nécessite une std::function<Tdst(Tsrc)> ....
        return TransformIterator<Iterator,std::function<Tdst(Tsrc)>,Tsrc,Tdst,Tdiff>(iterator, functor);
    }
}