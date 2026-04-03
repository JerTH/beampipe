use std::error::Error;
use std::fmt::Display;

use crate::ast::{Span, Sym};
use crate::token::TokenK;
use crate::value::Value;

#[derive(Debug, Clone)]
pub enum ParserErrorK {
    Unknown,
    Unexpected,
    UnexpectedToken { expected: TokenK, found: TokenK, span: Span },
    InvalidBinOp { span: Span },
    InvalidInfixOp { span: Span },
    InvalidLValue { span: Span },
    InvalidPrefixOp { span: Span },
    ExpectedExpression { context: &'static str, span: Span },
    ExpectedPath { context: &'static str, span: Span },
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

    pub fn single(kind: ParserErrorK) -> Self {
        Self {
            errs: vec![kind],
        }
    }
}

impl Display for ParserError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for err in &self.errs {
            writeln!(f, "{err:?}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ParserError {}

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
