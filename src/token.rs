use std::fmt::Display;

use crate::ast::{Expr, Span};

mod token_strs {
    pub const TOK_IF: &'static str = "if";
    pub const TOK_ELSE: &'static str = "else";
    pub const TOK_FN: &'static str = "fn";
    pub const TOK_LET: &'static str = "let";
    pub const TOK_MUT: &'static str = "mut";
    pub const TOK_REF: &'static str = "ref";
    pub const TOK_WHILE: &'static str = "while";
    pub const TOK_LOOP: &'static str = "loop";
    pub const TOK_RET: &'static str = "return";
    pub const TOK_TRUE: &'static str = "true";
    pub const TOK_FALSE: &'static str = "false";

    pub const TOK_L_PAREN: &'static str = "(";
    pub const TOK_R_PAREN: &'static str = ")";
    pub const TOK_L_BRACE: &'static str = "{";
    pub const TOK_R_BRACE: &'static str = "}";
    pub const TOK_L_BRACK: &'static str = "[";
    pub const TOK_R_BRACK: &'static str = "]";

    pub const TOK_ARROW: &'static str = "->";
    pub const TOK_PLUS: &'static str = "+";
    pub const TOK_MINUS: &'static str = "-";
    pub const TOK_EQ: &'static str = "=";
    pub const TOK_STAR: &'static str = "*";
    pub const TOK_SLASH: &'static str = "/";
    pub const TOK_COLON: &'static str = ":";
    pub const TOK_SEMI: &'static str = ";";
    pub const TOK_COMMA: &'static str = ",";
    pub const TOK_DOT: &'static str = ".";
    pub const TOK_LESS: &'static str = "<";
    pub const TOK_GREATER: &'static str = ">";

    pub const TOK_USIZE: &'static str = "usize";
    pub const TOK_ISIZE: &'static str = "isize";
    pub const TOK_I32: &'static str = "i32";
    pub const TOK_I64: &'static str = "i64";
    pub const TOK_U32: &'static str = "u32";
    pub const TOK_U64: &'static str = "u64";
    pub const TOK_F32: &'static str = "f32";
    pub const TOK_F64: &'static str = "f64";
}
pub use token_strs::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TokenK {
    KeyFn,
    KeyReturn,
    KeyLet,

    OpAdd,
    OpSub,
    OpMul,
    OpDiv,

    LitIdent,
    LitInt,
    LitFloat,
    LitBool,

    Ty,

    LParen,
    RParen,
    LBrack,
    RBrack,
    LBrace,
    RBrace,
    Semi,
    Dot,
    Arrow,
    Colon,
    Comma,

    KeyIf,
    KeyElse,
    KeyMut,
    KeyRef,
    KeyWhile,
    KeyLoop,
    Eof,
    ColCol,
    OpBang,
    OpEq,

    // comparison
    EqEq,
    OpLess,
    OpGreater,
}

impl Display for TokenK {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(&format!("{:?}", self))
    }
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenK,
    pub span: Span,
}

impl Token {
    pub fn new(span: Span, kind: TokenK) -> Self {
        Self { kind, span }
    }

    pub fn prefix_binding(&self) -> ((), usize) {
        match self.kind {
            TokenK::OpAdd => ((), 50),
            TokenK::OpSub => ((), 50),
            _ => ((), 00),
        }
    }

    pub fn infix_binding(&self) -> (usize, usize) {
        match self.kind {
            TokenK::OpAdd => (50, 50),
            TokenK::OpSub => (50, 50),
            TokenK::OpMul => (60, 60),
            TokenK::OpDiv => (60, 60),
            TokenK::OpEq => (90, 90),
            TokenK::Dot => (80, 80),
            TokenK::ColCol => (90, 90),
            TokenK::OpLess => (30, 30),
            TokenK::OpGreater => (30, 30),
            _ => (00, 00),
        }
    }

    pub fn postfix_binding(&self) -> (usize, ()) {
        match self.kind {
            _ => (0, ()),
        }
    }

    pub fn null_denotation(&self) -> fn() -> Expr {
        match self.kind {
            _ => || Expr::empty(),
        }
    }
}
