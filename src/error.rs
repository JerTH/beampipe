use std::error::Error;

use crate::ast::Sym;
use crate::codegen::Value;

#[derive(Debug, Clone)]
pub enum ParserErrorK {
    Unknown,
    Unexpected,
}

#[derive(Debug, Clone)]
pub struct ParserError {
    pub errs: Vec<ParserErrorK>,
}

impl ParserError {
    pub fn unknown() -> Self {
        Self {
            errs: vec![ParserErrorK::Unknown],
        }
    }
}

pub fn err_fatal<E: Error>(err: E, why: &'static str) -> ! {
    panic!("fatal internal error: {why},\n{err}")
}

pub fn err_op_mismatch(op: &'static str, lhs: Value, rhs: Value) -> ! {
    panic!("operator mismatch, cannot {op} {lhs:?} and {rhs:?}")
}

pub fn err_is_not<S: Into<String>>(value: &Value, kind: S) -> ! {
    panic!("{value:?} is not of type {:?}", kind.into())
}

pub fn err_sym_is_not<S: Into<String>>(sym: &Sym, kind: S) -> ! {
    panic!("{sym:?} is not a {:?}", kind.into())
}
