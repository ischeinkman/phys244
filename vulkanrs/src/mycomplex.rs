use std::ops::{Add, Div, Mul, Sub};

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct MyComplex {
    pub re: f32,
    pub im: f32,
}

impl MyComplex {
    #[inline(always)]
    pub const fn new(re: f32, im: f32) -> MyComplex {
        MyComplex { re, im }
    }
    #[inline(always)]
    pub const fn re(re: f32) -> MyComplex {
        MyComplex::new(re, 0.0)
    }
    #[inline(always)]
    pub const  fn im(im: f32) -> MyComplex {
        MyComplex::new(0.0, im)
    }
    #[inline(always)]
    pub const fn i() -> MyComplex {
        MyComplex::im(1.0)
    }
    #[inline(always)]
    pub const fn zero() -> MyComplex {
        MyComplex::re(0.0)
    }
    #[inline(always)]
    pub fn exp(self) -> MyComplex {
        let (sinim, cosim) = self.im.sin_cos();
        let expre = self.re.exp();
        MyComplex::new(expre * cosim, expre * sinim)
    }

    #[inline(always)]
    pub fn mag_2(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    #[inline(always)]
    pub fn to_bytes(self) -> [u8; 8] {
        let reb = self.re.to_le_bytes();
        let imb = self.im.to_le_bytes();
        [
            reb[0], reb[1], reb[2], reb[3], imb[0], imb[1], imb[2], imb[3],
        ]
    }

    #[inline(always)]
    pub fn from_bytes(bytes: [u8; 8]) -> MyComplex {
        let reb = [bytes[0], bytes[1], bytes[2], bytes[3]];
        let re = f32::from_le_bytes(reb);
        let imb = [bytes[4], bytes[5], bytes[6], bytes[7]];
        let im = f32::from_le_bytes(imb);
        MyComplex::new(re, im)
    }
}
impl Add<f32> for MyComplex {
    type Output = MyComplex;
    #[inline(always)]
    fn add(self, rhs: f32) -> MyComplex {
        MyComplex::new(self.re + rhs, self.im)
    }
}
impl Add<MyComplex> for MyComplex {
    type Output = MyComplex;
    #[inline(always)]
    fn add(self, other: MyComplex) -> MyComplex {
        MyComplex::new(self.re + other.re, self.im + other.im)
    }
}
impl Sub<MyComplex> for f32 {
    type Output = MyComplex;
    #[inline(always)]
    fn sub(self, complex_num: MyComplex) -> MyComplex {
        MyComplex::new(self - complex_num.re, -complex_num.im)
    }
}
impl Mul<f32> for MyComplex {
    type Output = MyComplex;
    #[inline(always)]
    fn mul(self, coeff: f32) -> MyComplex {
        MyComplex::new(self.re * coeff, self.im * coeff)
    }
}

impl Mul<MyComplex> for f32 {
    type Output = MyComplex;
    #[inline(always)]
    fn mul(self, num: MyComplex) -> MyComplex {
        MyComplex::new(num.re * self, num.im * self)
    }
}

impl Mul<MyComplex> for MyComplex {
    type Output = MyComplex;
    #[inline(always)]
    fn mul(self, other: MyComplex) -> MyComplex {
        let re = self.re * other.re - self.im * other.im;
        let im = self.im * other.re + self.re * other.im;
        MyComplex::new(re, im)
    }
}

impl Div<f32> for MyComplex {
    type Output = MyComplex;
    #[inline(always)]
    fn div(self, coeff_inv: f32) -> MyComplex {
        MyComplex::new(self.re / coeff_inv, self.im / coeff_inv)
    }
}
