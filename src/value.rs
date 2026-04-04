use std::{fmt::Display, ops::{Add, Div, Mul, Sub}, str::FromStr};

use crate::error::RuntimeErrorK;

#[derive(Debug, Default, Clone, PartialEq)]
pub enum Value {
    #[default]
    None,

    Int(i64),
    Float(f64),
    Bool(bool),
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::None => "None",
            Value::Int(_) => "Int",
            Value::Float(_) => "Float",
            Value::Bool(_) => "Bool",
        }
    }
}

impl Add for Value {
    type Output = Result<Value, RuntimeErrorK>;

    fn add(self, rhs: Self) -> Self::Output {
        match (&self, &rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
            _ => Err(RuntimeErrorK::TypeMismatch {
                op: "add",
                lhs: self.type_name(),
                rhs: rhs.type_name(),
            }),
        }
    }
}

impl Sub for Value {
    type Output = Result<Value, RuntimeErrorK>;

    fn sub(self, rhs: Self) -> Self::Output {
        match (&self, &rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
            _ => Err(RuntimeErrorK::TypeMismatch {
                op: "subtract",
                lhs: self.type_name(),
                rhs: rhs.type_name(),
            }),
        }
    }
}

impl Mul for Value {
    type Output = Result<Value, RuntimeErrorK>;

    fn mul(self, rhs: Self) -> Self::Output {
        match (&self, &rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
            _ => Err(RuntimeErrorK::TypeMismatch {
                op: "multiply",
                lhs: self.type_name(),
                rhs: rhs.type_name(),
            }),
        }
    }
}

impl Div for Value {
    type Output = Result<Value, RuntimeErrorK>;

    fn div(self, rhs: Self) -> Self::Output {
        match (&self, &rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a / b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),
            _ => Err(RuntimeErrorK::TypeMismatch {
                op: "divide",
                lhs: self.type_name(),
                rhs: rhs.type_name(),
            }),
        }
    }
}

impl Value {
    pub fn lt(self, rhs: Self) -> Result<Value, RuntimeErrorK> {
        match (&self, &rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(*a < *b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(*a < *b)),
            _ => Err(RuntimeErrorK::TypeMismatch {
                op: "compare (<)",
                lhs: self.type_name(),
                rhs: rhs.type_name(),
            }),
        }
    }

    pub fn gt(self, rhs: Self) -> Result<Value, RuntimeErrorK> {
        match (&self, &rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(*a > *b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(*a > *b)),
            _ => Err(RuntimeErrorK::TypeMismatch {
                op: "compare (>)",
                lhs: self.type_name(),
                rhs: rhs.type_name(),
            }),
        }
    }

    pub fn eq_val(self, rhs: Self) -> Result<Value, RuntimeErrorK> {
        match (&self, &rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(*a == *b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(*a == *b)),
            (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(*a == *b)),
            _ => Err(RuntimeErrorK::TypeMismatch {
                op: "compare (==)",
                lhs: self.type_name(),
                rhs: rhs.type_name(),
            }),
        }
    }

    pub fn neq(self, rhs: Self) -> Result<Value, RuntimeErrorK> {
        match (&self, &rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(*a != *b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(*a != *b)),
            (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(*a != *b)),
            _ => Err(RuntimeErrorK::TypeMismatch {
                op: "compare (!=)",
                lhs: self.type_name(),
                rhs: rhs.type_name(),
            }),
        }
    }

    pub fn and(self, rhs: Self) -> Result<Value, RuntimeErrorK> {
        match (&self, &rhs) {
            (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(*a && *b)),
            _ => Err(RuntimeErrorK::TypeMismatch {
                op: "logical and (&&)",
                lhs: self.type_name(),
                rhs: rhs.type_name(),
            }),
        }
    }

    pub fn or(self, rhs: Self) -> Result<Value, RuntimeErrorK> {
        match (&self, &rhs) {
            (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(*a || *b)),
            _ => Err(RuntimeErrorK::TypeMismatch {
                op: "logical or (||)",
                lhs: self.type_name(),
                rhs: rhs.type_name(),
            }),
        }
    }

    pub fn not(self) -> Result<Value, RuntimeErrorK> {
        match self {
            Value::Bool(a) => Ok(Value::Bool(!a)),
            _ => Err(RuntimeErrorK::UnaryTypeMismatch {
                op: "not (!)",
                operand: self.type_name(),
            }),
        }
    }

    pub fn neg(self) -> Result<Value, RuntimeErrorK> {
        match self {
            Value::Int(a) => Ok(Value::Int(-a)),
            Value::Float(a) => Ok(Value::Float(-a)),
            _ => Err(RuntimeErrorK::UnaryTypeMismatch {
                op: "negate (-)",
                operand: self.type_name(),
            }),
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
