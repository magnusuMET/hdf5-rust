use std::ffi::c_void;
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::ops::Deref;
use std::ptr;
use std::slice;

#[repr(C)]
pub struct VarLenArray<T: Copy> {
    len: usize,
    ptr: *const u8,
    tag: PhantomData<T>,
}

impl<T: Copy> VarLenArray<T> {
    pub unsafe fn from_parts(p: *const T, len: usize) -> Self {
        let (len, ptr) = if !p.is_null() && len != 0 {
            let dst = crate::malloc(len * mem::size_of::<T>());
            ptr::copy_nonoverlapping(p, dst.cast(), len);
            (len, dst)
        } else {
            (0, ptr::null_mut())
        };
        Self { len, ptr: ptr as *const _, tag: PhantomData }
    }

    #[inline]
    pub fn from_slice(arr: &[T]) -> Self {
        unsafe { Self::from_parts(arr.as_ptr(), arr.len()) }
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.cast()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len as _
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self
    }
}

impl<T: Copy> Drop for VarLenArray<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        unsafe {
            crate::free(self.ptr.cast_mut().cast());
        }
        self.ptr = ptr::null();
        self.len = 0;
    }
}

impl<T: Copy> Clone for VarLenArray<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self::from_slice(self)
    }
}

impl<T: Copy> Deref for VarLenArray<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        if self.len == 0 || self.ptr.is_null() {
            &[]
        } else {
            unsafe { slice::from_raw_parts(self.as_ptr(), self.len()) }
        }
    }
}

impl<'a, T: Copy> From<&'a [T]> for VarLenArray<T> {
    #[inline]
    fn from(arr: &[T]) -> Self {
        Self::from_slice(arr)
    }
}

impl<T: Copy> From<VarLenArray<T>> for Vec<T> {
    #[inline]
    fn from(v: VarLenArray<T>) -> Self {
        v.iter().copied().collect()
    }
}

impl<T: Copy, const N: usize> From<[T; N]> for VarLenArray<T> {
    #[inline]
    fn from(arr: [T; N]) -> Self {
        unsafe { Self::from_parts(arr.as_ptr(), arr.len()) }
    }
}

impl<T: Copy> Default for VarLenArray<T> {
    #[inline]
    fn default() -> Self {
        unsafe { Self::from_parts(ptr::null(), 0) }
    }
}

impl<T: Copy + PartialEq> PartialEq for VarLenArray<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Copy + Eq> Eq for VarLenArray<T> {}

impl<T: Copy + PartialEq> PartialEq<[T]> for VarLenArray<T> {
    #[inline]
    fn eq(&self, other: &[T]) -> bool {
        self.as_slice() == other
    }
}

impl<T: Copy + PartialEq, const N: usize> PartialEq<[T; N]> for VarLenArray<T> {
    #[inline]
    fn eq(&self, other: &[T; N]) -> bool {
        self.as_slice() == other
    }
}

impl<T: Copy + fmt::Debug> fmt::Debug for VarLenArray<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.as_slice().fmt(f)
    }
}

// Safety: Memory backed by `VarLenArray` can be accessed and freed from any thread
unsafe impl<T: Copy + Send> Send for VarLenArray<T> {}
// Safety: `VarLenArray` has no interior mutability
unsafe impl<T: Copy + Sync> Sync for VarLenArray<T> {}

/// Variant of VarLenArray which allows nested
/// derives of `H5Type`. This does not free memory
/// which must be done by the user.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct LeakyVarLenArray<T> {
    len: usize,
    ptr: *mut c_void,
    ph: PhantomData<*const T>,
}

impl<T> LeakyVarLenArray<T> {
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.cast()
    }
    #[inline]
    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr.cast()
    }
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        let len = self.len;
        if len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.as_ptr(), len) }
        }
    }
    #[inline]
    pub fn as_mut_slice(&self) -> &mut [T] {
        let len = self.len;
        if len == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), len) }
        }
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Drop this variable length array.
    /// OBS: Not called automatically
    pub fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }

        if std::mem::needs_drop::<T>() {
            for offset in 0..self.len() {
                unsafe {
                    std::ptr::drop_in_place(self.ptr.cast::<T>().offset(offset as isize));
                }
            }
        }

        unsafe { crate::free(self.ptr) }

        self.ptr = std::ptr::null_mut();
        self.len = 0;
    }
}

// Safety: Memory backed by `VarLenArray` can be accessed and freed from any thread
unsafe impl<T: Copy + Send> Send for LeakyVarLenArray<T> {}
// Safety: `VarLenArray` has no interior mutability
unsafe impl<T: Copy + Sync> Sync for LeakyVarLenArray<T> {}

#[cfg(test)]
pub mod tests {
    use super::{LeakyVarLenArray, VarLenArray};
    use crate::H5Type;

    type S = VarLenArray<u16>;

    #[test]
    pub fn test_vla_empty_default() {
        assert_eq!(&*S::default(), &[]);
        assert!(S::default().is_empty());
        assert_eq!(S::default().len(), 0);
    }

    #[test]
    pub fn test_vla_array_traits() {
        use std::slice;

        let s = &[1u16, 2, 3];
        let a = VarLenArray::from_slice(s);
        assert_eq!(a.as_slice(), s);
        assert_eq!(a.len(), 3);
        assert!(!a.is_empty());
        assert_eq!(unsafe { slice::from_raw_parts(a.as_ptr(), a.len()) }, s);
        assert_eq!(&*a, s);
        let c = a.clone();
        assert_eq!(&*a, &*c);
        let v: Vec<u16> = c.into();
        assert_eq!(v, vec![1, 2, 3]);
        assert_eq!(&*a, &*VarLenArray::from(*s));
        let f: [u16; 3] = [1, 2, 3];
        assert_eq!(&*a, &*VarLenArray::from(f));
        assert_eq!(format!("{:?}", a), "[1, 2, 3]");
        assert_eq!(a, [1, 2, 3]);
        assert_eq!(&a, s);
        assert_eq!(&a, a.as_slice());
        assert_eq!(a, a);
        let v: Vec<_> = a.iter().cloned().collect();
        assert_eq!(v, vec![1, 2, 3]);
    }

    #[test]
    fn impl_for_leaky_type() {
        type Stuff = LeakyVarLenArray<LeakyVarLenArray<LeakyVarLenArray<i32>>>;
        // #[repr(C)]
        // #[derive(Copy, Clone)]
        // struct Stuff {
        //     a: LeakyVarLenArray<LeakyVarLenArray<i32>>,
        // }

        println!("{:?}", <Stuff as H5Type>::type_descriptor());
    }
}
