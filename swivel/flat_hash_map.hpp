//          Copyright Malte Skarupke 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <cstdint>
#include <cstddef>
#include <functional>
#include <vector>
#include <cmath>
#include <array>
#include <algorithm>
#include <iterator>
#include <utility>
#include <type_traits>

#ifdef _MSC_VER
#define SKA_NOINLINE(...) __declspec(noinline) __VA_ARGS__
#else
#define SKA_NOINLINE(...) __VA_ARGS__ __attribute__((noinline))
#endif

namespace ska
{
struct prime_number_hash_policy;
struct power_of_two_hash_policy;

namespace detailv3
{
template<typename Result, typename Functor>
struct functor_storage : Functor
{
    functor_storage() = default;
    functor_storage(const Functor & functor)
        : Functor(functor)
    {
    }
    template<typename... Args>
    Result operator()(Args &&... args)
    {
        return static_cast<Functor &>(*this)(std::forward<Args>(args)...);
    }
    template<typename... Args>
    Result operator()(Args &&... args) const
    {
        return static_cast<const Functor &>(*this)(std::forward<Args>(args)...);
    }
};
template<typename Result, typename... Args>
struct functor_storage<Result, Result (*)(Args...)>
{
    typedef Result (*function_ptr)(Args...);
    function_ptr function;
    functor_storage(function_ptr function)
        : function(function)
    {
    }
    Result operator()(Args... args) const
    {
        return function(std::forward<Args>(args)...);
    }
    operator function_ptr &()
    {
        return function;
    }
    operator const function_ptr &()
    {
        return function;
    }
};
template<typename key_type, typename value_type, typename hasher>
struct KeyOrValueHasher : functor_storage<size_t, hasher>
{
    typedef functor_storage<size_t, hasher> hasher_storage;
    KeyOrValueHasher() = default;
    KeyOrValueHasher(const hasher & hash)
        : hasher_storage(hash)
    {
    }
    size_t operator()(const key_type & key)
    {
        return static_cast<hasher_storage &>(*this)(key);
    }
    size_t operator()(const key_type & key) const
    {
        return static_cast<const hasher_storage &>(*this)(key);
    }
    size_t operator()(const value_type & value)
    {
        return static_cast<hasher_storage &>(*this)(value.first);
    }
    size_t operator()(const value_type & value) const
    {
        return static_cast<const hasher_storage &>(*this)(value.first);
    }
    template<typename F, typename S>
    size_t operator()(const std::pair<F, S> & value)
    {
        return static_cast<hasher_storage &>(*this)(value.first);
    }
    template<typename F, typename S>
    size_t operator()(const std::pair<F, S> & value) const
    {
        return static_cast<const hasher_storage &>(*this)(value.first);
    }
};
template<typename key_type, typename value_type, typename key_equal>
struct KeyOrValueEquality : functor_storage<bool, key_equal>
{
    typedef functor_storage<bool, key_equal> equality_storage;
    KeyOrValueEquality() = default;
    KeyOrValueEquality(const key_equal & equality)
        : equality_storage(equality)
    {
    }
    bool operator()(const key_type & lhs, const key_type & rhs)
    {
        return static_cast<equality_storage &>(*this)(lhs, rhs);
    }
    bool operator()(const key_type & lhs, const value_type & rhs)
    {
        return static_cast<equality_storage &>(*this)(lhs, rhs.first);
    }
    bool operator()(const value_type & lhs, const key_type & rhs)
    {
        return static_cast<equality_storage &>(*this)(lhs.first, rhs);
    }
    bool operator()(const value_type & lhs, const value_type & rhs)
    {
        return static_cast<equality_storage &>(*this)(lhs.first, rhs.first);
    }
    template<typename F, typename S>
    bool operator()(const key_type & lhs, const std::pair<F, S> & rhs)
    {
        return static_cast<equality_storage &>(*this)(lhs, rhs.first);
    }
    template<typename F, typename S>
    bool operator()(const std::pair<F, S> & lhs, const key_type & rhs)
    {
        return static_cast<equality_storage &>(*this)(lhs.first, rhs);
    }
    template<typename F, typename S>
    bool operator()(const value_type & lhs, const std::pair<F, S> & rhs)
    {
        return static_cast<equality_storage &>(*this)(lhs.first, rhs.first);
    }
    template<typename F, typename S>
    bool operator()(const std::pair<F, S> & lhs, const value_type & rhs)
    {
        return static_cast<equality_storage &>(*this)(lhs.first, rhs.first);
    }
    template<typename FL, typename SL, typename FR, typename SR>
    bool operator()(const std::pair<FL, SL> & lhs, const std::pair<FR, SR> & rhs)
    {
        return static_cast<equality_storage &>(*this)(lhs.first, rhs.first);
    }
};
template<typename T>
struct sherwood_v3_entry
{
    static constexpr sherwood_v3_entry special_end_entry()
    {
        sherwood_v3_entry end;
        end.distance_from_desired = special_end_value;
        return end;
    }

    bool has_value() const
    {
        return distance_from_desired >= 0;
    }
    bool is_empty() const
    {
        return distance_from_desired < 0;
    }
    bool is_at_desired_position() const
    {
        return distance_from_desired <= 0;
    }
    template<typename... Args>
    void emplace(int8_t distance, Args &&... args)
    {
        new (&bytes) T(std::forward<Args>(args)...);
        distance_from_desired = distance;
    }

    void destroy_value()
    {
        value().~T();
        distance_from_desired = -1;
    }

    inline T & value()
    {
        return reinterpret_cast<T &>(bytes);
    }

    int8_t distance_from_desired = -1;
    static constexpr int8_t special_end_value = 0;
    typename std::aligned_storage<sizeof(T), alignof(T)>::type bytes = {};
};
static constexpr int8_t min_lookups = 4;
template<typename T>
struct EntryDefaultTable
{
    static constexpr const sherwood_v3_entry<T> table[min_lookups] =
    {
        {}, {}, {}, sherwood_v3_entry<T>::special_end_entry()
    };
};
template<typename T>
constexpr const sherwood_v3_entry<T> EntryDefaultTable<T>::table[min_lookups];

inline int8_t log2(size_t value)
{
    static constexpr int8_t table[64] =
    {
        63,  0, 58,  1, 59, 47, 53,  2,
        60, 39, 48, 27, 54, 33, 42,  3,
        61, 51, 37, 40, 49, 18, 28, 20,
        55, 30, 34, 11, 43, 14, 22,  4,
        62, 57, 46, 52, 38, 26, 32, 41,
        50, 36, 17, 19, 29, 10, 13, 21,
        56, 45, 25, 31, 35, 16,  9, 12,
        44, 24, 15,  8, 23,  7,  6,  5
    };
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value |= value >> 32;
    return table[((value - (value >> 1)) * 0x07EDD5E59A4E28C2) >> 58];
}
void throw_out_of_range();

template<typename T, bool>
struct AssignIfTrue
{
    void operator()(T & lhs, const T & rhs)
    {
        lhs = rhs;
    }
    void operator()(T & lhs, T && rhs)
    {
        lhs = std::move(rhs);
    }
};
template<typename T>
struct AssignIfTrue<T, false>
{
    void operator()(T &, const T &)
    {
    }
    void operator()(T &, T &&)
    {
    }
};

inline size_t next_power_of_two(size_t i)
{
    --i;
    i |= i >> 1;
    i |= i >> 2;
    i |= i >> 4;
    i |= i >> 8;
    i |= i >> 16;
    i |= i >> 32;
    ++i;
    return i;
}

template<typename...> using void_t = void;

template<typename T, typename = void>
struct HashPolicySelector
{
    typedef prime_number_hash_policy type;
};
template<typename T>
struct HashPolicySelector<T, void_t<typename T::hash_policy>>
{
    typedef typename T::hash_policy type;
};

template<typename T, typename FindKey, typename ArgumentHash, typename Hasher, typename ArgumentEqual, typename Equal, typename ArgumentAlloc, typename EntryAlloc>
class sherwood_v3_table : private EntryAlloc, private Hasher, private Equal
{
    using Entry = detailv3::sherwood_v3_entry<T>;
    using AllocatorTraits = std::allocator_traits<EntryAlloc>;
    using EntryPointer = typename AllocatorTraits::pointer;
    struct convertible_to_iterator;

public:

    using value_type = T;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using hasher = ArgumentHash;
    using key_equal = ArgumentEqual;
    using allocator_type = EntryAlloc;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;

    sherwood_v3_table()
    {
    }
    explicit sherwood_v3_table(size_type bucket_count, const ArgumentHash & hash = ArgumentHash(), const ArgumentEqual & equal = ArgumentEqual(), const ArgumentAlloc & alloc = ArgumentAlloc())
        : EntryAlloc(alloc), Hasher(hash), Equal(equal)
    {
        rehash(bucket_count);
    }
    sherwood_v3_table(size_type bucket_count, const ArgumentAlloc & alloc)
        : sherwood_v3_table(bucket_count, ArgumentHash(), ArgumentEqual(), alloc)
    {
    }
    sherwood_v3_table(size_type bucket_count, const ArgumentHash & hash, const ArgumentAlloc & alloc)
        : sherwood_v3_table(bucket_count, hash, ArgumentEqual(), alloc)
    {
    }
    explicit sherwood_v3_table(const ArgumentAlloc & alloc)
        : EntryAlloc(alloc)
    {
    }
    template<typename It>
    sherwood_v3_table(It first, It last, size_type bucket_count = 0, const ArgumentHash & hash = ArgumentHash(), const ArgumentEqual & equal = ArgumentEqual(), const ArgumentAlloc & alloc = ArgumentAlloc())
        : sherwood_v3_table(bucket_count, hash, equal, alloc)
    {
        insert(first, last);
    }
    template<typename It>
    sherwood_v3_table(It first, It last, size_type bucket_count, const ArgumentAlloc & alloc)
        : sherwood_v3_table(first, last, bucket_count, ArgumentHash(), ArgumentEqual(), alloc)
    {
    }
    template<typename It>
    sherwood_v3_table(It first, It last, size_type bucket_count, const ArgumentHash & hash, const ArgumentAlloc & alloc)
        : sherwood_v3_table(first, last, bucket_count, hash, ArgumentEqual(), alloc)
    {
    }
    sherwood_v3_table(std::initializer_list<T> il, size_type bucket_count = 0, const ArgumentHash & hash = ArgumentHash(), const ArgumentEqual & equal = ArgumentEqual(), const ArgumentAlloc & alloc = ArgumentAlloc())
        : sherwood_v3_table(bucket_count, hash, equal, alloc)
    {
        if (bucket_count == 0)
            rehash(il.size());
        insert(il.begin(), il.end());
    }
    sherwood_v3_table(std::initializer_list<T> il, size_type bucket_count, const ArgumentAlloc & alloc)
        : sherwood_v3_table(il, bucket_count, ArgumentHash(), ArgumentEqual(), alloc)
    {
    }
    sherwood_v3_table(std::initializer_list<T> il, size_type bucket_count, const ArgumentHash & hash, const ArgumentAlloc & alloc)
        : sherwood_v3_table(il, bucket_count, hash, ArgumentEqual(), alloc)
    {
    }
    sherwood_v3_table(const sherwood_v3_table & other)
        : sherwood_v3_table(other, AllocatorTraits::select_on_container_copy_construction(other.get_allocator()))
    {
    }
    sherwood_v3_table(const sherwood_v3_table & other, const ArgumentAlloc & alloc)
        : EntryAlloc(alloc), Hasher(other), Equal(other), _max_load_factor(other._max_load_factor)
    {
        rehash_for_other_container(other);
        try
        {
            insert(other.begin(), other.end());
        }
        catch(...)
        {
            clear();
            deallocate_data(entries, num_slots_minus_one, max_lookups);
            throw;
        }
    }
    sherwood_v3_table(sherwood_v3_table && other) noexcept
        : EntryAlloc(std::move(other)), Hasher(std::move(other)), Equal(std::move(other))
    {
        swap_pointers(other);
    }
    sherwood_v3_table(sherwood_v3_table && other, const ArgumentAlloc & alloc) noexcept
        : EntryAlloc(alloc), Hasher(std::move(other)), Equal(std::move(other))
    {
        swap_pointers(other);
    }
    sherwood_v3_table & operator=(const sherwood_v3_table & other)
    {
        if (this == std::addressof(other))
            return *this;

        clear();
        if (AllocatorTraits::propagate_on_container_copy_assignment::value)
        {
            if (static_cast<EntryAlloc &>(*this) != static_cast<const EntryAlloc &>(other))
            {
                reset_to_empty_state();
            }
            AssignIfTrue<EntryAlloc, AllocatorTraits::propagate_on_container_copy_assignment::value>()(*this, other);
        }
        _max_load_factor = other._max_load_factor;
        static_cast<Hasher &>(*this) = other;
        static_cast<Equal &>(*this) = other;
        rehash_for_other_container(other);
        insert(other.begin(), other.end());
        return *this;
    }
    sherwood_v3_table & operator=(sherwood_v3_table && other) noexcept
    {
        if (this == std::addressof(other))
            return *this;
        else if (AllocatorTraits::propagate_on_container_move_assignment::value)
        {
            clear();
            reset_to_empty_state();
            AssignIfTrue<EntryAlloc, AllocatorTraits::propagate_on_container_move_assignment::value>()(*this, std::move(other));
            swap_pointers(other);
        }
        else if (static_cast<EntryAlloc &>(*this) == static_cast<EntryAlloc &>(other))
        {
            swap_pointers(other);
        }
        else
        {
            clear();
            _max_load_factor = other._max_load_factor;
            rehash_for_other_container(other);
            for (T & elem : other)
                emplace(std::move(elem));
            other.clear();
        }
        static_cast<Hasher &>(*this) = std::move(other);
        static_cast<Equal &>(*this) = std::move(other);
        return *this;
    }
    ~sherwood_v3_table()
    {
        clear();
        deallocate_data(entries, num_slots_minus_one, max_lookups);
    }

    const allocator_type & get_allocator() const
    {
        return static_cast<const allocator_type &>(*this);
    }
    const ArgumentEqual & key_eq() const
    {
        return static_cast<const ArgumentEqual &>(*this);
    }
    const ArgumentHash & hash_function() const
    {
        return static_cast<const ArgumentHash &>(*this);
    }

    template<typename ValueType>
    struct templated_iterator
    {
        EntryPointer current = EntryPointer();

        using iterator_category = std::forward_iterator_tag;
        using value_type = ValueType;
        using difference_type = ptrdiff_t;
        using pointer = ValueType *;
        using reference = ValueType &;

        friend bool operator==(const templated_iterator & lhs, const templated_iterator & rhs)
        {
            return lhs.current == rhs.current;
        }
        friend bool operator!=(const templated_iterator & lhs, const templated_iterator & rhs)
        {
            return !(lhs == rhs);
        }

        templated_iterator & operator++()
        {
            do
            {
                ++current;
            }
            while(current->is_empty());
            return *this;
        }
        templated_iterator operator++(int)
        {
            templated_iterator copy(*this);
            ++*this;
            return copy;
        }

        ValueType & operator*() const
        {
            return current->value();
        }
        ValueType * operator->() const
        {
            return std::addressof(current->value());
        }

        operator templated_iterator<const value_type>() const
        {
            return { current };
        }
    };
    using iterator = templated_iterator<value_type>;
    using const_iterator = templated_iterator<const value_type>;

    iterator begin()
    {
        for (EntryPointer it = entries;; ++it)
        {
            if (it->has_value())
                return { it };
        }
    }
    const_iterator begin() const
    {
        for (EntryPointer it = entries;; ++it)
        {
            if (it->has_value())
                return { it };
        }
    }
    const_iterator cbegin() const
    {
        return begin();
    }
    iterator end()
    {
        return { entries + static_cast<ptrdiff_t>(num_slots_minus_one + max_lookups) };
    }
    const_iterator end() const
    {
        return { entries + static_cast<ptrdiff_t>(num_slots_minus_one + max_lookups) };
    }
    const_iterator cend() const
    {
        return end();
    }

    iterator find(const FindKey & key)
    {
        size_t index = hash_policy.index_for_hash(hash_object(key), num_slots_minus_one);
        EntryPointer it = entries + ptrdiff_t(index);
        for (int8_t distance = 0;; ++it, ++distance)
        {
            if (it->distance_from_desired < distance)
                return end();
            else if (compares_equal(key, it->value()))
                return { it };
        }
    }
    const_iterator find(const FindKey & key) const
    {
        return const_cast<sherwood_v3_table *>(this)->find(key);
    }
    size_t count(const FindKey & key) const
    {
        return find(key) == end() ? 0 : 1;
    }
    std::pair<iterator, iterator> equal_range(const FindKey & key)
    {
        iterator found = find(key);
        if (found == end())
            return { found, found };
        else
            return { found, std::next(found) };
    }
    std::pair<const_iterator, const_iterator> equal_range(const FindKey & key) const
    {
        const_iterator found = find(key);
        if (found == end())
            return { found, found };
        else
            return { found, std::next(found) };
    }

    template<typename Key, typename... Args>
    std::pair<iterator, bool> emplace(Key && key, Args &&... args)
    {
        size_t index = hash_policy.index_for_hash(hash_object(key), num_slots_minus_one);
        EntryPointer current_entry = entries + ptrdiff_t(index);
        int8_t distance_from_desired = 0;
        for (;; ++distance_from_desired, ++current_entry)
        {
            if (current_entry->distance_from_desired < distance_from_desired)
                break;
            else if (compares_equal(key, current_entry->value()))
                return { { current_entry }, false };
        }
        return emplace_new_key(distance_from_desired, current_entry, std::forward<Key>(key), std::forward<Args>(args)...);
    }

    std::pair<iterator, bool> insert(const value_type & value)
    {
        return emplace(value);
    }
    std::pair<iterator, bool> insert(value_type && value)
    {
        return emplace(std::move(value));
    }
    template<typename... Args>
    iterator emplace_hint(const_iterator, Args &&... args)
    {
        return emplace(std::forward<Args>(args)...).first;
    }
    iterator insert(const_iterator, const value_type & value)
    {
        return emplace(value).first;
    }
    iterator insert(const_iterator, value_type && value)
    {
        return emplace(std::move(value)).first;
    }

    template<typename It>
    void insert(It begin, It end)
    {
        for (; begin != end; ++begin)
        {
            emplace(*begin);
        }
    }
    void insert(std::initializer_list<value_type> il)
    {
        insert(il.begin(), il.end());
    }

    void rehash(size_t num_buckets)
    {
        num_buckets = std::max(num_buckets, static_cast<size_t>(std::ceil(num_elements / static_cast<double>(_max_load_factor))));
        if (num_buckets == 0)
        {
            reset_to_empty_state();
            return;
        }
        auto new_prime_index = hash_policy.next_size_over(num_buckets);
        if (num_buckets == bucket_count())
            return;
        int8_t new_max_lookups = compute_max_lookups(num_buckets);
        EntryPointer new_buckets(AllocatorTraits::allocate(*this, num_buckets + new_max_lookups));
        for (EntryPointer it = new_buckets, real_end = it + static_cast<ptrdiff_t>(num_buckets + new_max_lookups - 1); it != real_end; ++it)
        {
            it->distance_from_desired = -1;
        }
        new_buckets[num_buckets + new_max_lookups - 1].distance_from_desired = Entry::special_end_value;
        std::swap(entries, new_buckets);
        std::swap(num_slots_minus_one, num_buckets);
        --num_slots_minus_one;
        hash_policy.commit(new_prime_index);
        int8_t old_max_lookups = max_lookups;
        max_lookups = new_max_lookups;
        num_elements = 0;
        for (EntryPointer it = new_buckets, end = it + static_cast<ptrdiff_t>(num_buckets + old_max_lookups); it != end; ++it)
        {
            if (it->has_value())
            {
                emplace(std::move(it->value()));
                it->destroy_value();
            }
        }
        deallocate_data(new_buckets, num_buckets, old_max_lookups);
    }

    void reserve(size_t num_elements)
    {
        size_t required_buckets = num_buckets_for_reserve(num_elements);
        if (required_buckets > bucket_count())
            rehash(required_buckets);
    }

    // the return value is a type that can be converted to an iterator
    // the reason for doing this is that it's not free to find the
    // iterator pointing at the next element. if you care about the
    // next iterator, turn the return value into an iterator
    convertible_to_iterator erase(const_iterator to_erase)
    {
        EntryPointer current = to_erase.current;
        current->destroy_value();
        --num_elements;
        for (EntryPointer next = current + ptrdiff_t(1); !next->is_at_desired_position(); ++current, ++next)
        {
            current->emplace(next->distance_from_desired - 1, std::move(next->value()));
            next->destroy_value();
        }
        return { to_erase.current };
    }

    iterator erase(const_iterator begin_it, const_iterator end_it)
    {
        for (EntryPointer it = begin_it.current, end = end_it.current; it != end; ++it)
        {
            if (it->has_value())
            {
                it->destroy_value();
                --num_elements;
            }
        }
        if (end_it == this->end())
            return this->end();
        ptrdiff_t num_to_move = std::min(static_cast<ptrdiff_t>(end_it.current->distance_from_desired), end_it.current - begin_it.current);
        EntryPointer to_return = end_it.current - num_to_move;
        for (EntryPointer it = end_it.current; !it->is_at_desired_position();)
        {
            EntryPointer target = it - num_to_move;
            target->emplace(it->distance_from_desired - num_to_move, std::move(it->value()));
            it->destroy_value();
            ++it;
            num_to_move = std::min(static_cast<ptrdiff_t>(it->distance_from_desired), num_to_move);
        }
        return { to_return };
    }

    size_t erase(const FindKey & key)
    {
        auto found = find(key);
        if (found == end())
            return 0;
        else
        {
            erase(found);
            return 1;
        }
    }

    void clear()
    {
        for (EntryPointer it = entries, end = it + static_cast<ptrdiff_t>(num_slots_minus_one + max_lookups); it != end; ++it)
        {
            if (it->has_value())
                it->destroy_value();
        }
        num_elements = 0;
    }

    void shrink_to_fit()
    {
        rehash_for_other_container(*this);
    }

    void swap(sherwood_v3_table & other)
    {
        using std::swap;
        swap_pointers(other);
        swap(static_cast<ArgumentHash &>(*this), static_cast<ArgumentHash &>(other));
        swap(static_cast<ArgumentEqual &>(*this), static_cast<ArgumentEqual &>(other));
        if (AllocatorTraits::propagate_on_container_swap::value)
            swap(static_cast<EntryAlloc &>(*this), static_cast<EntryAlloc &>(other));
    }

    size_t size() const
    {
        return num_elements;
    }
    size_t max_size() const
    {
        return (AllocatorTraits::max_size(*this)) / sizeof(Entry);
    }
    size_t bucket_count() const
    {
        return num_slots_minus_one + 1;
    }
    size_type max_bucket_count() const
    {
        return (AllocatorTraits::max_size(*this) - min_lookups) / sizeof(Entry);
    }
    size_t bucket(const FindKey & key) const
    {
        return hash_policy.index_for_hash(hash_object(key), num_slots_minus_one);
    }
    float load_factor() const
    {
        size_t buckets = bucket_count();
        if (buckets)
            return static_cast<float>(num_elements) / bucket_count();
        else
            return 0;
    }
    void max_load_factor(float value)
    {
        _max_load_factor = value;
    }
    float max_load_factor() const
    {
        return _max_load_factor;
    }

    bool empty() const
    {
        return num_elements == 0;
    }

private:
    typename HashPolicySelector<ArgumentHash>::type hash_policy;
    int8_t max_lookups = detailv3::min_lookups - 1;
    float _max_load_factor = 0.5f;
    using DefaultTable = detailv3::EntryDefaultTable<T>;
    EntryPointer entries = const_cast<Entry *>(DefaultTable::table);
    size_t num_slots_minus_one = 0;
    size_t num_elements = 0;

    static int8_t compute_max_lookups(size_t num_buckets)
    {
        int8_t desired = detailv3::log2(num_buckets);
        return std::max(detailv3::min_lookups, desired);
    }

    size_t num_buckets_for_reserve(size_t num_elements) const
    {
        return static_cast<size_t>(std::ceil(num_elements / std::min(0.5, static_cast<double>(_max_load_factor))));
    }
    void rehash_for_other_container(const sherwood_v3_table & other)
    {
        rehash(std::min(num_buckets_for_reserve(other.size()), other.bucket_count()));
    }

    void swap_pointers(sherwood_v3_table & other)
    {
        using std::swap;
        swap(hash_policy, other.hash_policy);
        swap(entries, other.entries);
        swap(num_slots_minus_one, other.num_slots_minus_one);
        swap(num_elements, other.num_elements);
        swap(max_lookups, other.max_lookups);
        swap(_max_load_factor, other._max_load_factor);
    }

    template<typename Key, typename... Args>
    SKA_NOINLINE(std::pair<iterator, bool>) emplace_new_key(int8_t distance_from_desired, EntryPointer current_entry, Key && key, Args &&... args)
    {
        using std::swap;
        if (num_slots_minus_one == 0 || distance_from_desired == max_lookups || static_cast<double>(num_elements + 1) / static_cast<double>(bucket_count()) > _max_load_factor)
        {
            grow();
            return emplace(std::forward<Key>(key), std::forward<Args>(args)...);
        }
        else if (current_entry->is_empty())
        {
            current_entry->emplace(distance_from_desired, std::forward<Key>(key), std::forward<Args>(args)...);
            ++num_elements;
            return { { current_entry }, true };
        }
        value_type to_insert(std::forward<Key>(key), std::forward<Args>(args)...);
        swap(distance_from_desired, current_entry->distance_from_desired);
        swap(to_insert, current_entry->value());
        iterator result = { current_entry };
        for (++distance_from_desired, ++current_entry;; ++current_entry)
        {
            if (current_entry->is_empty())
            {
                current_entry->emplace(distance_from_desired, std::move(to_insert));
                ++num_elements;
                return { result, true };
            }
            else if (current_entry->distance_from_desired < distance_from_desired)
            {
                swap(distance_from_desired, current_entry->distance_from_desired);
                swap(to_insert, current_entry->value());
                ++distance_from_desired;
            }
            else
            {
                ++distance_from_desired;
                if (distance_from_desired == max_lookups)
                {
                    swap(to_insert, result.current->value());
                    grow();
                    return emplace(std::move(to_insert));
                }
            }
        }
    }

    void grow()
    {
        rehash(std::max(size_t(4), 2 * bucket_count()));
    }

    void deallocate_data(EntryPointer begin, size_t num_slots_minus_one, int8_t max_lookups)
    {
        if (begin != const_cast<Entry *>(DefaultTable::table))
        {
            AllocatorTraits::deallocate(*this, begin, num_slots_minus_one + max_lookups + 1);
        }
    }

    void reset_to_empty_state()
    {
        deallocate_data(entries, num_slots_minus_one, max_lookups);
        entries = const_cast<Entry *>(DefaultTable::table);
        num_slots_minus_one = 0;
        hash_policy.reset();
        max_lookups = detailv3::min_lookups - 1;
    }

    template<typename U>
    size_t hash_object(const U & key)
    {
        return static_cast<Hasher &>(*this)(key);
    }
    template<typename U>
    size_t hash_object(const U & key) const
    {
        return static_cast<const Hasher &>(*this)(key);
    }
    template<typename L, typename R>
    bool compares_equal(const L & lhs, const R & rhs)
    {
        return static_cast<Equal &>(*this)(lhs, rhs);
    }

    struct convertible_to_iterator
    {
        EntryPointer it;

        operator iterator()
        {
            if (it->has_value())
                return { it };
            else
                return ++iterator{it};
        }
        operator const_iterator()
        {
            if (it->has_value())
                return { it };
            else
                return ++const_iterator{it};
        }
    };

};
}

struct prime_number_hash_policy
{
    size_t index_for_hash(size_t hash, size_t /*num_slots_minus_one*/) const
    {
        switch(prime_index)
        {
        case 0:
            return 0llu;
        case 1:
            return hash % 2llu;
        case 2:
            return hash % 3llu;
        case 3:
            return hash % 5llu;
        case 4:
            return hash % 7llu;
        case 5:
            return hash % 11llu;
        case 6:
            return hash % 13llu;
        case 7:
            return hash % 17llu;
        case 8:
            return hash % 23llu;
        case 9:
            return hash % 29llu;
        case 10:
            return hash % 37llu;
        case 11:
            return hash % 47llu;
        case 12:
            return hash % 59llu;
        case 13:
            return hash % 73llu;
        case 14:
            return hash % 97llu;
        case 15:
            return hash % 127llu;
        case 16:
            return hash % 151llu;
        case 17:
            return hash % 197llu;
        case 18:
            return hash % 251llu;
        case 19:
            return hash % 313llu;
        case 20:
            return hash % 397llu;
        case 21:
            return hash % 499llu;
        case 22:
            return hash % 631llu;
        case 23:
            return hash % 797llu;
        case 24:
            return hash % 1009llu;
        case 25:
            return hash % 1259llu;
        case 26:
            return hash % 1597llu;
        case 27:
            return hash % 2011llu;
        case 28:
            return hash % 2539llu;
        case 29:
            return hash % 3203llu;
        case 30:
            return hash % 4027llu;
        case 31:
            return hash % 5087llu;
        case 32:
            return hash % 6421llu;
        case 33:
            return hash % 8089llu;
        case 34:
            return hash % 10193llu;
        case 35:
            return hash % 12853llu;
        case 36:
            return hash % 16193llu;
        case 37:
            return hash % 20399llu;
        case 38:
            return hash % 25717llu;
        case 39:
            return hash % 32401llu;
        case 40:
            return hash % 40823llu;
        case 41:
            return hash % 51437llu;
        case 42:
            return hash % 64811llu;
        case 43:
            return hash % 81649llu;
        case 44:
            return hash % 102877llu;
        case 45:
            return hash % 129607llu;
        case 46:
            return hash % 163307llu;
        case 47:
            return hash % 205759llu;
        case 48:
            return hash % 259229llu;
        case 49:
            return hash % 326617llu;
        case 50:
            return hash % 411527llu;
        case 51:
            return hash % 518509llu;
        case 52:
            return hash % 653267llu;
        case 53:
            return hash % 823117llu;
        case 54:
            return hash % 1037059llu;
        case 55:
            return hash % 1306601llu;
        case 56:
            return hash % 1646237llu;
        case 57:
            return hash % 2074129llu;
        case 58:
            return hash % 2613229llu;
        case 59:
            return hash % 3292489llu;
        case 60:
            return hash % 4148279llu;
        case 61:
            return hash % 5226491llu;
        case 62:
            return hash % 6584983llu;
        case 63:
            return hash % 8296553llu;
        case 64:
            return hash % 10453007llu;
        case 65:
            return hash % 13169977llu;
        case 66:
            return hash % 16593127llu;
        case 67:
            return hash % 20906033llu;
        case 68:
            return hash % 26339969llu;
        case 69:
            return hash % 33186281llu;
        case 70:
            return hash % 41812097llu;
        case 71:
            return hash % 52679969llu;
        case 72:
            return hash % 66372617llu;
        case 73:
            return hash % 83624237llu;
        case 74:
            return hash % 105359939llu;
        case 75:
            return hash % 132745199llu;
        case 76:
            return hash % 167248483llu;
        case 77:
            return hash % 210719881llu;
        case 78:
            return hash % 265490441llu;
        case 79:
            return hash % 334496971llu;
        case 80:
            return hash % 421439783llu;
        case 81:
            return hash % 530980861llu;
        case 82:
            return hash % 668993977llu;
        case 83:
            return hash % 842879579llu;
        case 84:
            return hash % 1061961721llu;
        case 85:
            return hash % 1337987929llu;
        case 86:
            return hash % 1685759167llu;
        case 87:
            return hash % 2123923447llu;
        case 88:
            return hash % 2675975881llu;
        case 89:
            return hash % 3371518343llu;
        case 90:
            return hash % 4247846927llu;
        case 91:
            return hash % 5351951779llu;
        case 92:
            return hash % 6743036717llu;
        case 93:
            return hash % 8495693897llu;
        case 94:
            return hash % 10703903591llu;
        case 95:
            return hash % 13486073473llu;
        case 96:
            return hash % 16991387857llu;
        case 97:
            return hash % 21407807219llu;
        case 98:
            return hash % 26972146961llu;
        case 99:
            return hash % 33982775741llu;
        case 100:
            return hash % 42815614441llu;
        case 101:
            return hash % 53944293929llu;
        case 102:
            return hash % 67965551447llu;
        case 103:
            return hash % 85631228929llu;
        case 104:
            return hash % 107888587883llu;
        case 105:
            return hash % 135931102921llu;
        case 106:
            return hash % 171262457903llu;
        case 107:
            return hash % 215777175787llu;
        case 108:
            return hash % 271862205833llu;
        case 109:
            return hash % 342524915839llu;
        case 110:
            return hash % 431554351609llu;
        case 111:
            return hash % 543724411781llu;
        case 112:
            return hash % 685049831731llu;
        case 113:
            return hash % 863108703229llu;
        case 114:
            return hash % 1087448823553llu;
        case 115:
            return hash % 1370099663459llu;
        case 116:
            return hash % 1726217406467llu;
        case 117:
            return hash % 2174897647073llu;
        case 118:
            return hash % 2740199326961llu;
        case 119:
            return hash % 3452434812973llu;
        case 120:
            return hash % 4349795294267llu;
        case 121:
            return hash % 5480398654009llu;
        case 122:
            return hash % 6904869625999llu;
        case 123:
            return hash % 8699590588571llu;
        case 124:
            return hash % 10960797308051llu;
        case 125:
            return hash % 13809739252051llu;
        case 126:
            return hash % 17399181177241llu;
        case 127:
            return hash % 21921594616111llu;
        case 128:
            return hash % 27619478504183llu;
        case 129:
            return hash % 34798362354533llu;
        case 130:
            return hash % 43843189232363llu;
        case 131:
            return hash % 55238957008387llu;
        case 132:
            return hash % 69596724709081llu;
        case 133:
            return hash % 87686378464759llu;
        case 134:
            return hash % 110477914016779llu;
        case 135:
            return hash % 139193449418173llu;
        case 136:
            return hash % 175372756929481llu;
        case 137:
            return hash % 220955828033581llu;
        case 138:
            return hash % 278386898836457llu;
        case 139:
            return hash % 350745513859007llu;
        case 140:
            return hash % 441911656067171llu;
        case 141:
            return hash % 556773797672909llu;
        case 142:
            return hash % 701491027718027llu;
        case 143:
            return hash % 883823312134381llu;
        case 144:
            return hash % 1113547595345903llu;
        case 145:
            return hash % 1402982055436147llu;
        case 146:
            return hash % 1767646624268779llu;
        case 147:
            return hash % 2227095190691797llu;
        case 148:
            return hash % 2805964110872297llu;
        case 149:
            return hash % 3535293248537579llu;
        case 150:
            return hash % 4454190381383713llu;
        case 151:
            return hash % 5611928221744609llu;
        case 152:
            return hash % 7070586497075177llu;
        case 153:
            return hash % 8908380762767489llu;
        case 154:
            return hash % 11223856443489329llu;
        case 155:
            return hash % 14141172994150357llu;
        case 156:
            return hash % 17816761525534927llu;
        case 157:
            return hash % 22447712886978529llu;
        case 158:
            return hash % 28282345988300791llu;
        case 159:
            return hash % 35633523051069991llu;
        case 160:
            return hash % 44895425773957261llu;
        case 161:
            return hash % 56564691976601587llu;
        case 162:
            return hash % 71267046102139967llu;
        case 163:
            return hash % 89790851547914507llu;
        case 164:
            return hash % 113129383953203213llu;
        case 165:
            return hash % 142534092204280003llu;
        case 166:
            return hash % 179581703095829107llu;
        case 167:
            return hash % 226258767906406483llu;
        case 168:
            return hash % 285068184408560057llu;
        case 169:
            return hash % 359163406191658253llu;
        case 170:
            return hash % 452517535812813007llu;
        case 171:
            return hash % 570136368817120201llu;
        case 172:
            return hash % 718326812383316683llu;
        case 173:
            return hash % 905035071625626043llu;
        case 174:
            return hash % 1140272737634240411llu;
        case 175:
            return hash % 1436653624766633509llu;
        case 176:
            return hash % 1810070143251252131llu;
        case 177:
            return hash % 2280545475268481167llu;
        case 178:
            return hash % 2873307249533267101llu;
        case 179:
            return hash % 3620140286502504283llu;
        case 180:
            return hash % 4561090950536962147llu;
        case 181:
            return hash % 5746614499066534157llu;
        case 182:
            return hash % 7240280573005008577llu;
        case 183:
            return hash % 9122181901073924329llu;
        case 184:
            return hash % 11493228998133068689llu;
        case 185:
            return hash % 14480561146010017169llu;
        case 186:
            return hash % 18446744073709551557llu;
        default:
            return hash;
        }
    }
    uint8_t next_size_over(size_t & size) const
    {
        // prime numbers generated by the following method:
        // 1. start with a prime p = 2
        // 2. go to wolfram alpha and get p = NextPrime(2 * p)
        // 3. repeat 2. until you overflow 64 bits
        // you now have large gaps which you would hit if somebody called reserve() with an unlucky number.
        // 4. to fill the gaps for every prime p go to wolfram alpha and get ClosestPrime(p * 2^(1/3)) and ClosestPrime(p * 2^(2/3)) and put those in the gaps
        // 5. get PrevPrime(2^64) and put it at the end
        static constexpr const size_t prime_list[] =
        {
            2llu, 3llu, 5llu, 7llu, 11llu, 13llu, 17llu, 23llu, 29llu, 37llu, 47llu,
            59llu, 73llu, 97llu, 127llu, 151llu, 197llu, 251llu, 313llu, 397llu,
            499llu, 631llu, 797llu, 1009llu, 1259llu, 1597llu, 2011llu, 2539llu,
            3203llu, 4027llu, 5087llu, 6421llu, 8089llu, 10193llu, 12853llu, 16193llu,
            20399llu, 25717llu, 32401llu, 40823llu, 51437llu, 64811llu, 81649llu,
            102877llu, 129607llu, 163307llu, 205759llu, 259229llu, 326617llu,
            411527llu, 518509llu, 653267llu, 823117llu, 1037059llu, 1306601llu,
            1646237llu, 2074129llu, 2613229llu, 3292489llu, 4148279llu, 5226491llu,
            6584983llu, 8296553llu, 10453007llu, 13169977llu, 16593127llu, 20906033llu,
            26339969llu, 33186281llu, 41812097llu, 52679969llu, 66372617llu,
            83624237llu, 105359939llu, 132745199llu, 167248483llu, 210719881llu,
            265490441llu, 334496971llu, 421439783llu, 530980861llu, 668993977llu,
            842879579llu, 1061961721llu, 1337987929llu, 1685759167llu, 2123923447llu,
            2675975881llu, 3371518343llu, 4247846927llu, 5351951779llu, 6743036717llu,
            8495693897llu, 10703903591llu, 13486073473llu, 16991387857llu,
            21407807219llu, 26972146961llu, 33982775741llu, 42815614441llu,
            53944293929llu, 67965551447llu, 85631228929llu, 107888587883llu,
            135931102921llu, 171262457903llu, 215777175787llu, 271862205833llu,
            342524915839llu, 431554351609llu, 543724411781llu, 685049831731llu,
            863108703229llu, 1087448823553llu, 1370099663459llu, 1726217406467llu,
            2174897647073llu, 2740199326961llu, 3452434812973llu, 4349795294267llu,
            5480398654009llu, 6904869625999llu, 8699590588571llu, 10960797308051llu,
            13809739252051llu, 17399181177241llu, 21921594616111llu, 27619478504183llu,
            34798362354533llu, 43843189232363llu, 55238957008387llu, 69596724709081llu,
            87686378464759llu, 110477914016779llu, 139193449418173llu,
            175372756929481llu, 220955828033581llu, 278386898836457llu,
            350745513859007llu, 441911656067171llu, 556773797672909llu,
            701491027718027llu, 883823312134381llu, 1113547595345903llu,
            1402982055436147llu, 1767646624268779llu, 2227095190691797llu,
            2805964110872297llu, 3535293248537579llu, 4454190381383713llu,
            5611928221744609llu, 7070586497075177llu, 8908380762767489llu,
            11223856443489329llu, 14141172994150357llu, 17816761525534927llu,
            22447712886978529llu, 28282345988300791llu, 35633523051069991llu,
            44895425773957261llu, 56564691976601587llu, 71267046102139967llu,
            89790851547914507llu, 113129383953203213llu, 142534092204280003llu,
            179581703095829107llu, 226258767906406483llu, 285068184408560057llu,
            359163406191658253llu, 452517535812813007llu, 570136368817120201llu,
            718326812383316683llu, 905035071625626043llu, 1140272737634240411llu,
            1436653624766633509llu, 1810070143251252131llu, 2280545475268481167llu,
            2873307249533267101llu, 3620140286502504283llu, 4561090950536962147llu,
            5746614499066534157llu, 7240280573005008577llu, 9122181901073924329llu,
            11493228998133068689llu, 14480561146010017169llu, 18446744073709551557llu
        };
        const size_t * found = std::lower_bound(std::begin(prime_list), std::end(prime_list) - 1, size);
        size = *found;
        return static_cast<uint8_t>(1 + found - prime_list);
    }
    void commit(uint8_t new_prime_index)
    {
        prime_index = new_prime_index;
    }
    void reset()
    {
        prime_index = 0;
    }

private:
    uint8_t prime_index = 0;
};

struct power_of_two_hash_policy
{
    size_t index_for_hash(size_t hash, size_t num_slots_minus_one) const
    {
        return hash & num_slots_minus_one;
    }
    int8_t next_size_over(size_t & size) const
    {
        size = detailv3::next_power_of_two(size);
        return 0;
    }
    void commit(int8_t)
    {
    }
    void reset()
    {
    }

};

template<typename K, typename V, typename H = std::hash<K>, typename E = std::equal_to<K>, typename A = std::allocator<std::pair<K, V> > >
class flat_hash_map
        : public detailv3::sherwood_v3_table
        <
            std::pair<K, V>,
            K,
            H,
            detailv3::KeyOrValueHasher<K, std::pair<K, V>, H>,
            E,
            detailv3::KeyOrValueEquality<K, std::pair<K, V>, E>,
            A,
            typename std::allocator_traits<A>::template rebind_alloc<detailv3::sherwood_v3_entry<std::pair<K, V>>>
        >
{
    using Table = detailv3::sherwood_v3_table
    <
        std::pair<K, V>,
        K,
        H,
        detailv3::KeyOrValueHasher<K, std::pair<K, V>, H>,
        E,
        detailv3::KeyOrValueEquality<K, std::pair<K, V>, E>,
        A,
        typename std::allocator_traits<A>::template rebind_alloc<detailv3::sherwood_v3_entry<std::pair<K, V>>>
    >;
public:

    using key_type = K;
    using mapped_type = V;

    using Table::Table;
    flat_hash_map()
    {
    }

    V & operator[](const K & key)
    {
        return emplace(key, convertible_to_value()).first->second;
    }
    V & operator[](K && key)
    {
        return emplace(std::move(key), convertible_to_value()).first->second;
    }
    V & at(const K & key)
    {
        auto found = this->find(key);
        if (found == this->end())
            throw std::out_of_range("Argument passed to at() was not in the map.");
        return found->second;
    }
    const V & at(const K & key) const
    {
        auto found = this->find(key);
        if (found == this->end())
            throw std::out_of_range("Argument passed to at() was not in the map.");
        return found->second;
    }

    using Table::emplace;
    std::pair<typename Table::iterator, bool> emplace()
    {
        return emplace(key_type(), convertible_to_value());
    }

    friend bool operator==(const flat_hash_map & lhs, const flat_hash_map & rhs)
    {
        if (lhs.size() != rhs.size())
            return false;
        for (const typename Table::value_type & value : lhs)
        {
            auto found = rhs.find(value.first);
            if (found == rhs.end())
                return false;
            else if (value.second != found->second)
                return false;
        }
        return true;
    }
    friend bool operator!=(const flat_hash_map & lhs, const flat_hash_map & rhs)
    {
        return !(lhs == rhs);
    }

private:
    struct convertible_to_value
    {
        operator V() const
        {
            return V();
        }
    };
};

template<typename T, typename H = std::hash<T>, typename E = std::equal_to<T>, typename A = std::allocator<T> >
class flat_hash_set
        : public detailv3::sherwood_v3_table
        <
            T,
            T,
            H,
            detailv3::functor_storage<size_t, H>,
            E,
            detailv3::functor_storage<bool, E>,
            A,
            typename std::allocator_traits<A>::template rebind_alloc<detailv3::sherwood_v3_entry<T>>
        >
{
    using Table = detailv3::sherwood_v3_table
    <
        T,
        T,
        H,
        detailv3::functor_storage<size_t, H>,
        E,
        detailv3::functor_storage<bool, E>,
        A,
        typename std::allocator_traits<A>::template rebind_alloc<detailv3::sherwood_v3_entry<T>>
    >;
public:

    using key_type = T;

    using Table::Table;
    flat_hash_set()
    {
    }

    template<typename... Args>
    std::pair<typename Table::iterator, bool> emplace(Args &&... args)
    {
        return Table::emplace(T(std::forward<Args>(args)...));
    }
    std::pair<typename Table::iterator, bool> emplace(const key_type & arg)
    {
        return Table::emplace(arg);
    }
    std::pair<typename Table::iterator, bool> emplace(key_type & arg)
    {
        return Table::emplace(arg);
    }
    std::pair<typename Table::iterator, bool> emplace(const key_type && arg)
    {
        return Table::emplace(std::move(arg));
    }
    std::pair<typename Table::iterator, bool> emplace(key_type && arg)
    {
        return Table::emplace(std::move(arg));
    }

    friend bool operator==(const flat_hash_set & lhs, const flat_hash_set & rhs)
    {
        if (lhs.size() != rhs.size())
            return false;
        for (const T & value : lhs)
        {
            if (rhs.find(value) == rhs.end())
                return false;
        }
        return true;
    }
    friend bool operator!=(const flat_hash_set & lhs, const flat_hash_set & rhs)
    {
        return !(lhs == rhs);
    }
};


template<typename T>
struct power_of_two_std_hash : std::hash<T>
{
    typedef ska::power_of_two_hash_policy hash_policy;
};

} // end namespace ska
