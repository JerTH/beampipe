use std::{cell::RefCell, fmt::Display};

use crate::ast::{Ident, Sym};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Marker {
    Symbol(Sym),
    Ident(Ident),
    Offset(isize),
    Temporary,
}

impl Display for Marker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Marker::Ident(ident) => write!(f, "{}", ident),
            Marker::Offset(offset) => {
                let sig_str = match offset.signum() {
                    0 => "",
                    1 => "+",
                    -1 => "-",
                    _ => panic!("invalid signum"),
                };
                write!(f, "{}{}", sig_str, offset)
            }
            Marker::Symbol(sym) => write!(f, "&{}", sym.as_string()),
            Marker::Temporary => write!(f, "&TEMP_PLACEHOLDER"),
        }
    }
}

/// The product of the first step of lowering the AST
#[derive(Debug, PartialOrd)]
pub enum Ir {
    Nop,
    Add,
    Sub,
    Div,
    Mul,

    CmpLess,
    CmpGreater,

    Bool(bool),
    Integer(i64),
    Float(f64),

    Jump(Marker),
    JumpTrue(Marker),
    JumpFalse(Marker),

    Load(Marker),
    Store(Marker),

    // A symbol - usually preceeds a function def
    Symbol(Sym),

    Call(Sym),
    Return,
}

impl PartialEq for Ir {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bool(l0), Self::Bool(r0)) => l0 == r0,
            (Self::Integer(l0), Self::Integer(r0)) => l0 == r0,
            (Self::Float(l0), Self::Float(r0)) => {
                assert!(!(l0.is_nan() && r0.is_nan()));
                l0 == r0
            }
            (Self::Jump(l0), Self::Jump(r0)) => l0 == r0,
            (Self::JumpTrue(l0), Self::JumpTrue(r0)) => l0 == r0,
            (Self::JumpFalse(l0), Self::JumpFalse(r0)) => l0 == r0,
            (Self::Load(l0), Self::Load(r0)) => l0 == r0,
            (Self::Store(l0), Self::Store(r0)) => l0 == r0,
            (Self::Symbol(l0), Self::Symbol(r0)) => l0 == r0,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl Eq for Ir {}

impl Display for Ir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ir::Symbol(_) => {}
            _ => {
                write!(f, "\t")?;
            }
        }

        match self {
            Ir::Nop => write!(f, "NOP"),
            Ir::Add => write!(f, "ADD"),
            Ir::Sub => write!(f, "SUB"),
            Ir::Div => write!(f, "DIV"),
            Ir::Mul => write!(f, "MUL"),
            Ir::Store(marker) => write!(f, "STORE({})", marker),
            Ir::Load(marker) => write!(f, "LOAD({})", marker),
            Ir::Bool(value) => write!(f, "BOOL({})", value),
            Ir::Integer(value) => write!(f, "INT({})", value),
            Ir::Float(value) => write!(f, "FLOAT({})", value),
            Ir::Jump(marker) => write!(f, "JMP({})", marker),
            Ir::JumpTrue(marker) => write!(f, "JMPC({})", marker),
            Ir::JumpFalse(marker) => write!(f, "JMPN({})", marker),
            Ir::Symbol(sym) => write!(f, "{}", sym),
            Ir::CmpLess => write!(f, "LT"),
            Ir::CmpGreater => write!(f, "GT"),
            Ir::Return => write!(f, "RET"),
            Ir::Call(sym) => write!(f, "CALL({})", sym),
        }
    }
}

#[derive(Debug, Default)]
pub struct IrCode {
    code: RefCell<Vec<Ir>>,
}

impl Display for IrCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (index, item) in self.code.borrow().iter().enumerate() {
            writeln!(f, "{index:<4} {item}")?;
        }
        Ok(())
    }
}

impl IrCode {
    /// Emit a new code at the end of the stream
    pub fn emit(&self, ir: Ir) {
        self.code.borrow_mut().push(ir);
    }

    /// Change an already emitted code at a specified location if
    /// the code already at that location matches `pred`
    ///
    /// Useful for patching markers
    pub fn patch(&self, ir: Ir, at: usize, pred: Ir) {
        assert!(at < self.len());

        if let Some(code) = self.code.borrow_mut().get_mut(at) {
            if *code == pred {
                *code = ir
            } else {
                panic!(
                    "codegen: expected {:?} but found {:?} during patch operation",
                    pred, code
                );
            }
        }
    }

    pub fn len(&self) -> usize {
        self.code.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.code.borrow().is_empty()
    }
}
