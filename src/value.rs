use std::{fmt::Display, ops::{Add, Div, Mul, Sub}, str::FromStr};

#[derive(Debug, Default, Clone, PartialEq)]
pub enum Value {
    #[default]
    None,

    Int(i64),
    Float(f64),
    Bool(bool),
}

impl Add for Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a + b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a + b),
            _ => {
                unimplemented!()
            }
        }
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a - b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a - b),
            _ => {
                unimplemented!()
            }
        }
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a * b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a * b),
            _ => {
                unimplemented!()
            }
        }
    }
}

impl Div for Value {
    type Output = Value;

    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a / b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a / b),
            _ => {
                unimplemented!()
            }
        }
    }
}

impl Value {
    pub fn lt(self, rhs: Self) -> Value {
        match (self, rhs) {
            (Value::Int(a), Value::Int(b)) => Value::Bool(a < b),
            (Value::Float(a), Value::Float(b)) => Value::Bool(a < b),
            (_a, _b) => {
                unimplemented!()
            }
        }
    }

    pub fn gt(self, rhs: Self) -> Value {
        match (self, rhs) {
            (Value::Int(a), Value::Int(b)) => Value::Bool(a > b),
            (Value::Float(a), Value::Float(b)) => Value::Bool(a > b),
            (_a, _b) => {
                unimplemented!()
            }
        }
    }

    pub fn eq_val(self, rhs: Self) -> Value {
        match (self, rhs) {
            (Value::Int(a), Value::Int(b)) => Value::Bool(a == b),
            (Value::Float(a), Value::Float(b)) => Value::Bool(a == b),
            (Value::Bool(a), Value::Bool(b)) => Value::Bool(a == b),
            (_a, _b) => {
                unimplemented!()
            }
        }
    }

    pub fn neq(self, rhs: Self) -> Value {
        match (self, rhs) {
            (Value::Int(a), Value::Int(b)) => Value::Bool(a != b),
            (Value::Float(a), Value::Float(b)) => Value::Bool(a != b),
            (Value::Bool(a), Value::Bool(b)) => Value::Bool(a != b),
            (_a, _b) => {
                unimplemented!()
            }
        }
    }

    pub fn and(self, rhs: Self) -> Value {
        match (self, rhs) {
            (Value::Bool(a), Value::Bool(b)) => Value::Bool(a && b),
            (_a, _b) => {
                unimplemented!()
            }
        }
    }

    pub fn or(self, rhs: Self) -> Value {
        match (self, rhs) {
            (Value::Bool(a), Value::Bool(b)) => Value::Bool(a || b),
            (_a, _b) => {
                unimplemented!()
            }
        }
    }

    pub fn not(self) -> Value {
        match self {
            Value::Bool(a) => Value::Bool(!a),
            _ => {
                unimplemented!()
            }
        }
    }

    pub fn neg(self) -> Value {
        match self {
            Value::Int(a) => Value::Int(-a),
            Value::Float(a) => Value::Float(-a),
            _ => {
                unimplemented!()
            }
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::None => write!(f, "()"),
            Value::Int(value) => write!(f, "{value}"),
            Value::Float(value) => write!(f, "{value}"),
            Value::Bool(value) => write!(f, "{value:?}"),
        }
    }
}

impl FromStr for Value {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Ok(value) = s.parse::<i64>() {
            Ok(Value::Int(value))
        } else if let Ok(value) = s.parse::<f64>() {
            Ok(Value::Float(value))
        } else {
            Err(())
        }
    }
}
