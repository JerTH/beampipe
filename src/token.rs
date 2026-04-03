use std::fmt::Display;

use crate::ast::Span;

mod token_strs {
    pub const TOK_IF: &str = "if";
    pub const TOK_ELSE: &str = "else";
    pub const TOK_FN: &str = "fn";
    pub const TOK_LET: &str = "let";
    pub const TOK_MUT: &str = "mut";
    pub const TOK_REF: &str = "ref";
    pub const TOK_WHILE: &str = "while";
    pub const TOK_LOOP: &str = "loop";
    pub const TOK_RET: &str = "return";
    pub const TOK_TRUE: &str = "true";
    pub const TOK_FALSE: &str = "false";

    pub const TOK_L_PAREN: &str = "(";
    pub const TOK_R_PAREN: &str = ")";
    pub const TOK_L_BRACE: &str = "{";
    pub const TOK_R_BRACE: &str = "}";
    pub const TOK_L_BRACK: &str = "[";
    pub const TOK_R_BRACK: &str = "]";

    pub const TOK_ARROW: &str = "->";
    pub const TOK_PLUS: &str = "+";
    pub const TOK_MINUS: &str = "-";
    pub const TOK_EQ: &str = "=";
    pub const TOK_STAR: &str = "*";
    pub const TOK_SLASH: &str = "/";
    pub const TOK_COLON: &str = ":";
    pub const TOK_SEMI: &str = ";";
    pub const TOK_COMMA: &str = ",";
    pub const TOK_DOT: &str = ".";
    pub const TOK_LESS: &str = "<";
    pub const TOK_GREATER: &str = ">";

    pub const TOK_USIZE: &str = "usize";
    pub const TOK_ISIZE: &str = "isize";
    pub const TOK_I32: &str = "i32";
    pub const TOK_I64: &str = "i64";
    pub const TOK_U32: &str = "u32";
    pub const TOK_U64: &str = "u64";
    pub const TOK_F32: &str = "f32";
    pub const TOK_F64: &str = "f64";
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
            TokenK::OpEq => (2, 1),         // lowest precedence, right-associative
            TokenK::OpLess => (30, 31),
            TokenK::OpGreater => (30, 31),
            TokenK::OpAdd => (50, 51),
            TokenK::OpSub => (50, 51),
            TokenK::OpMul => (60, 61),
            TokenK::OpDiv => (60, 61),
            TokenK::Dot => (80, 81),
            TokenK::ColCol => (90, 91),
            _ => (0, 0),
        }
    }

    pub fn postfix_binding(&self) -> (usize, ()) {
        (0, ())
    }

}
