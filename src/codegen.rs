use core::panic;
use std::{cell::RefCell, collections::HashMap, fmt::Display, ops::{Add, Deref, Div, Mul, Sub}, str::FromStr};

use crate::ast::{*, Ident};


#[derive(Debug, Default, Clone)]
pub enum Value {
    #[default]
    None,

    Int(i64),
    Float(f64),
    Ident(Ident),
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
    fn lt(self, rhs: Self) -> Value {
        match (self, rhs) {
            (Value::Int(a), Value::Int(b)) => Value::Bool(a < b),
            (Value::Float(a), Value::Float(b)) => Value::Bool(a < b),
            (_a, _b) => {
                unimplemented!()
            }
        }
    }

    fn gt(self, rhs: Self) -> Value {
        match (self, rhs) {
            (Value::Int(a), Value::Int(b)) => Value::Bool(a > b),
            (Value::Float(a), Value::Float(b)) => Value::Bool(a > b),
            (_a, _b) => {
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
            Value::Ident(value) => write!(f, "{value:?}"),
            Value::Bool(value) => write!(f, "{value:?}"),
        }
    }
}

impl FromStr for Value {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Ok(value) = s.parse::<i64>() {
            return Ok(Value::Int(value))
        } else if let Ok(value) = s.parse::<f64>() {
            return Ok(Value::Float(value))
        } else {
            Err(())
        }
    }
}

pub struct Eval {
    vars: RefCell<HashMap<String, Value>>,
    func: RefCell<HashMap<String, Ptr<Fn>>>,
    symt: RefCell<SymTable>,
}

impl Eval {
    pub fn eval(expr: &Expr) -> Value {
        println!("Evaluating...");

        let state = Self {
            vars: Default::default(),
            symt: Default::default(),
            func: Default::default(),
        };
        state.eval_r(expr)
    }

    fn eval_r(&self, expr: &Expr) -> Value {
        let (_, _) = (expr.astid, expr.span);
        
        match &expr.kind {
            ExprK::Empty => {
                return Value::None
            },
            ExprK::Semi(expr) => {
                self.eval_r(expr);
                Value::None
            },
            ExprK::Local(_) => todo!(),
            ExprK::Item(_item) => {
                todo!()
            },
            ExprK::Lit(lit) => {
                match &lit.kind {
                    LitK::Int => return Value::Int(lit.symbol.parse::<i64>().expect("expected i64 to parse")),
                    LitK::Float => return Value::Float(lit.symbol.parse::<f64>().expect("expected f64 to parse")),
                    LitK::Bool => return Value::Bool(lit.symbol.parse::<bool>().expect("expected boolean to parse")),
                }
            },
            ExprK::Block(block) => {
                let mut returns = Value::None;
                for expr in &block.list {
                    returns = self.eval_r(expr);
                }
                if let Value::Ident(ident) = &returns {
                    if let Some(value) = self.vars.borrow().get(&ident.as_string()) {
                        returns = value.clone()
                    }
                }
                returns
            },
            ExprK::Assign(lhs, rhs) => {
                match (self.eval_r(lhs), self.eval_r(rhs)) {
                    (Value::Ident(ident), value) => {
                        self.vars
                            .borrow_mut()
                            .entry(ident.as_string())
                            .and_modify(|v| *v = value.clone())
                            .or_insert(value);
                    },
                    (lhs, _) => {
                        err_is_not(&lhs, "identifier");
                    },
                }
                return Value::None;
            },
            ExprK::BinOp(op, lhs, rhs) => {
                match op.kind {
                    BinOpK::Add => {
                        self.eval_r(lhs).add(self.eval_r(rhs))
                    },
                    BinOpK::Sub => {
                        self.eval_r(lhs).sub(self.eval_r(rhs))
                    },
                    BinOpK::Div => {
                        self.eval_r(lhs).div(self.eval_r(rhs))
                    },
                    BinOpK::Mul => {
                        self.eval_r(lhs).mul(self.eval_r(rhs))
                    },
                    BinOpK::CmpLess => {
                        self.eval_r(lhs).gt(self.eval_r(rhs))
                    },
                    BinOpK::CmpGreater => {
                        self.eval_r(lhs).gt(self.eval_r(rhs))
                    },
                }
            },
            ExprK::AssignOp(_, _, _) => {
                todo!()
            },
            ExprK::Path(path) => {
                let full_name: String = path.list.iter().fold(String::new(), |acc, x| {
                    format!("{}::{}", acc, x)
                })
                .trim_start_matches("::")
                .into();

                let ident = Ident {
                    name: self.symt.borrow().make(full_name),
                    span: Span::none(),
                };

                if let Some(value) = self.vars.borrow().get(&ident.as_string()) {
                    return value.clone()
                } else {
                    return Value::Ident(ident)
                }
            },
            ExprK::Fn(func) => {
                let path = func.path.clone();
                self.func
                    .borrow_mut()
                    .entry(String::from(&path))
                    .and_modify(|_| {
                        panic!("multiple declarations of {:?}", path)
                    })
                    .or_insert(func.clone());
                
                // main entry-point for eval
                if let Some(fn_path_name) = path.list.last() {
                    if fn_path_name.as_string().as_str() == "main" {
                        let path_ptr = Ptr::new(path);

                        let main_args = Vec::new();
                        return self.call_fn(&path_ptr, &main_args);
                    }
                }
                
                Value::None
            },
            ExprK::If(cond, then, else_then) => {
                match self.eval_r(cond) {
                    Value::Bool(true) => {
                        return self.eval_r(then)
                    },
                    Value::Bool(false) => {
                        if let Some(expr) = else_then {
                            return self.eval_r(expr)
                        } else {
                            return Value::None
                        }
                    },
                    condr => {
                        err_is_not(&condr, "boolean");
                    }
                }
            },
            ExprK::Call(name, args) => {
                match name.kind {
                    ExprK::Path(ref fnpath) => {
                        self.call_fn(fnpath, args)
                    }
                    _ => {
                        panic!("fn name is not a path");
                    }
                }
            },
            ExprK::Loop(loop_block) => {
                match &loop_block.kind {
                    LoopK::For(head, body) => {
                        todo!()
                    },
                    LoopK::While(cond, body) => {
                        todo!()
                    },
                    LoopK::Loop(label, body) => {
                        todo!()
                    },
                }
            },
        }
    }

    fn call_fn(&self, func: &Ptr<Path>, args: &Vec<Ptr<FnArg>>) -> Value {
        let func_path_string = String::from(func);

        if let Some(func) = self.func.borrow().get(&func_path_string) {
            for (param, arg) in func.sig.params.iter().zip(args.iter()) {
                let resolved_param = self.eval_r(&arg.expr);
                self.vars.borrow_mut().insert(param.ident.as_string(), resolved_param);
            }
            return self.eval_r(&func.body)
        } else {
            let funcs_clone = self.func.borrow().clone();
            let funcs = funcs_clone.iter().map(|f| f.0).collect::<Vec<_>>();
            panic!("undeclared fn\n{:#?}\n{:#?}\n", func, funcs);
        }
    }
}

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
                    _ => panic!("invalid signum")
                };
                write!(f, "{}{}", sig_str, offset)
            },
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
                assert_eq!(false, l0.is_nan() && r0.is_nan());
                l0 == r0
            },
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
            Ir::Symbol(_) => {},
            _ => { write!(f, "\t")?; },
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
                panic!("codegen: expected {:?} but found {:?} during patch operation", pred, code);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.code.borrow().len()
    }
}

#[derive(Default)]
pub struct Emit {
    code: IrCode,
    syms: SymTable,
}

impl Emit {
    pub fn emit(expr: &Expr) -> IrCode {
        let mut emit = Self::default();
        Emit::emit_r(&mut emit, expr);
        return emit.code;
    }

    fn emit_r(&self, expr: &Expr) {
        let (_, _) = (expr.astid, expr.span);
        let code = &self.code;
        let syms = &self.syms;
        
        match &expr.kind {
            ExprK::Empty => {
                code.emit(Ir::Nop)
            },

            ExprK::Semi(expr) => {
                self.emit_r(expr)
            },
            
            ExprK::Local(_) => todo!(),
            ExprK::Item(_) => todo!(),
            ExprK::Lit(lit) => {
                match lit.kind {
                    LitK::Bool => code.emit(Ir::Bool(lit.symbol.parse::<bool>().unwrap_or_else(|_| err_sym_is_not(&lit.symbol, "boolean")))),
                    LitK::Int => code.emit(Ir::Integer(lit.symbol.parse::<i64>().unwrap_or_else(|_| err_sym_is_not(&lit.symbol, "integer")))),
                    LitK::Float => code.emit(Ir::Float(lit.symbol.parse::<f64>().unwrap_or_else(|_| err_sym_is_not(&lit.symbol, "float")))),
                }
            },
            ExprK::Block(block) => {
                for expr in &block.list {
                    self.emit_r(expr);
                }
            },
            ExprK::Assign(lhs, rhs) => {
                match Eval::eval(lhs) {
                    Value::Ident(ident) => {
                        self.emit_r(rhs);
                        code.emit(Ir::Store(Marker::Ident(ident)));
                    },
                    value => err_is_not(&value, "identifier"),
                }
            },
            ExprK::BinOp(op, lhs, rhs) => {
                {
                    self.emit_r(lhs);
                    self.emit_r(rhs);
                }
                
                match op.kind {
                    BinOpK::Add => { code.emit(Ir::Add) },
                    BinOpK::Sub => { code.emit(Ir::Sub) },
                    BinOpK::Div => { code.emit(Ir::Div) },
                    BinOpK::Mul => { code.emit(Ir::Mul) },
                    BinOpK::CmpLess => { code.emit(Ir::CmpLess) },
                    BinOpK::CmpGreater => { code.emit(Ir::CmpGreater) },
                }
            },
            ExprK::AssignOp(_, _, _) => {
                todo!()
            },
            ExprK::Path(_path) => {
                match Eval::eval(&expr) {
                    Value::None => {},
                    Value::Int(_value) => todo!(),
                    Value::Float(_value) => todo!(),
                    Value::Ident(ident) => {
                        code.emit(Ir::Load(Marker::Ident(ident)))
                    },
                    Value::Bool(_value) => todo!(),
                }
            },
            // A function declaration
            ExprK::Fn(func) => {
                let path = &func.path;
                let symbol = self.syms.make(path);
                self.code.emit(Ir::Symbol(symbol));

                if let ExprK::Block(body) = &func.body.kind {
                    for expr in &body.list {
                        self.emit_r(expr);
                    }
                } else {
                    panic!("fn body is not a code block")
                }

                self.code.emit(Ir::Return);
            },
            
            ExprK::If(pred, blk, else_blk) => {
                self.emit_r(pred);
                
                let cond_location = self.code.len();
                self.code.emit(Ir::JumpFalse(Marker::Temporary));
                
                self.emit_r(blk);

                let mut cond_jump_offset = self.code.len() as isize - cond_location as isize;
                
                if let Some(else_block) = else_blk {
                    let jump_else_location = self.code.len();
                    self.code.emit(Ir::Jump(Marker::Temporary));
                    self.emit_r(&else_block);
                    
                    let else_jump_offset = self.code.len() as isize - jump_else_location as isize;
                    self.code.patch(Ir::Jump(Marker::Offset(else_jump_offset)), jump_else_location, Ir::Jump(Marker::Temporary));

                    cond_jump_offset += 1;
                }

                self.code.patch(Ir::JumpFalse(Marker::Offset(cond_jump_offset)), cond_location, Ir::JumpFalse(Marker::Temporary));
                //todo!("codegen for if-else statement");
            },
            ExprK::Call(path, args) => {
                match &path.kind {
                    ExprK::Path(fnpath) => {
                        let fn_path_string = String::from(fnpath);
                        let sym = self.syms.make(fn_path_string);
                        self.code.emit(Ir::Call(sym));
                    },
                    _ => {
                        panic!("call path is not a path");
                    }
                }
            },
            ExprK::Loop(loop_block) => {
                match &loop_block.kind {
                    LoopK::For(head, body) => {
                        todo!()
                    },
                    LoopK::While(cond, body) => {
                        todo!()
                    },
                    LoopK::Loop(label, body) => {
                        todo!()
                    },
                }
            },
        }
    }
}



#[cfg(test)]
mod test {
    use crate::parse::Parse;
    use super::*;

    #[test]
    fn test_parse_expressions() {
        let source = include_str!("example/expr.rs");
        let expr = Parse::parse(source).expect("expected ast");
        let code = Emit::emit(&expr);
        let result = Eval::eval(&expr);
        
        //println!("{:#?}", expr);
        println!("{}", code);
        println!("{}", source);
        println!("{}", result);
    }

    #[test]
    fn test_parse_branching() {
        std::env::set_var("RUST_BACKTRACE", "1");

        let source = include_str!("example/foo.rs");
        let expr = Parse::parse(source).expect("expected ast");
        
        let code = Emit::emit(&expr);
        let result = Eval::eval(&expr);
        
        println!("{:#?}", expr);
        println!("{}", code);
        println!("{}", source);
        println!("{}", result);
    }

    #[test]
    fn test_parse_nesting() {
        std::env::set_var("RUST_BACKTRACE", "1");

        let source = include_str!("example/nest.rs");
        let expr = Parse::parse(source).expect("expected ast");

        let code = Emit::emit(&expr);
        let result = Eval::eval(&expr);
        
        println!("{:#?}", expr);
        println!("{}", code);
        println!("{}", source);
        println!("{}", result);
    }
}

fn err_op_mismatch(op: &'static str, lhs: Value, rhs: Value) -> ! {
    panic!("operator mismatch, cannot {op} {lhs:?} and {rhs:?}")
}

fn err_is_not<S: Into<String>>(value: &Value, kind: S) -> ! {
    panic!("{value:?} is not of type {:?}", kind.into())
}

fn err_sym_is_not<S: Into<String>>(sym: &Sym, kind: S) -> ! {
    panic!("{sym:?} is not a {:?}", kind.into())
}

//fn err_unexpected_char(unexpected: char) -> ! {
//    panic!("unexpected character: {unexpected}");
//}
//
//fn err_malformed_token(buffer: &Vec<char>) -> ! {
//    let string = String::from_iter(buffer.into_iter());
//    panic!("malformed token: {string}");
//}
