#![allow(dead_code)]
#![allow(unused_variables)]

use std::{sync::Arc, collections::HashMap, ops::Deref};

type Ptr<T> = Arc<T>;

#[derive(Debug)]
pub enum Expr {
    Block(Ptr<Block>),

    Conditional {
        condition: Ptr<Expr>,
        then_expr: Ptr<Block>,
        else_expr: Ptr<Block>,
    },
    Assign {
        lhs: Ptr<Expr>,
        rhs: Ptr<Expr>,
    },
    Add {
        lhs: Ptr<Expr>,
        rhs: Ptr<Expr>,
    },
    Sub {
        lhs: Ptr<Expr>,
        rhs: Ptr<Expr>,
    },
    FunctionDef {
        ident: Ident,
        param: Vec<(Param, Value)>,
        body: Block,
    },
    FunctionCall {
        ident: Ident,
        param: Vec<(Param, Value)>,
    },
    Value(Value),
}

#[derive(Debug)]
pub struct Ty(u64);

#[derive(Debug)]
pub struct Block(Vec<Ptr<Expr>>);
impl Deref for Block {
    type Target = Vec<Ptr<Expr>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug)]
pub struct Param {
    ident: Ident,
    ty: Ty,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ident(String);

impl Ident {
    fn prefix<S: Into<String>>(&self, prefix: S) -> Self {
        Ident(String::from(format!("{}_{}", prefix.into(), self.0)))
    }
}

impl From<&'static str> for Ident { fn from(value: &'static str) -> Self { Ident(String::from(value)) } }
impl From<String> for Ident { fn from(value: String) -> Self { Ident(value) } }
impl From<&String> for Ident { fn from(value: &String) -> Self { Ident(value.clone()) } }
impl From<Ident> for String { fn from(value: Ident) -> Self { value.0 } }
impl From<&Ident> for String { fn from(value: &Ident) -> Self { value.0.clone() } }

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Ident(Ident),
    None,
}

/// Constructors
impl Expr {
    pub fn conditional(condition: Ptr<Expr>, then_expr: Ptr<Block>, else_expr: Ptr<Block>) -> Ptr<Self> {
        Ptr::new(Self::Conditional {
            condition,
            then_expr,
            else_expr,
        })
    }

    pub fn add(lhs: Ptr<Expr>, rhs: Ptr<Expr>) -> Ptr<Self> {
        Ptr::new(Self::Add {
            lhs,
            rhs,
        })
    }

    pub fn value(value: Value) -> Arc<Self> {
        Ptr::new(Expr::Value(value))
    }

    pub fn block<I: IntoIterator<Item = Expr>>(iter: I) -> Self {
        Expr::Block(Ptr::new(Block(Vec::from_iter(iter.into_iter().map(|expr| Arc::new(expr))))))
    }
}

/// Constructors
impl Value {
    pub fn int(value: i64) -> Self {
        Value::Int(value)
    }

    pub fn float(value: f64) -> Self {
        Value::Float(value)
    }
}

/// Operations
impl Value {
    pub fn add(lhs: Self, rhs: Self) -> Self {
        match (lhs, rhs) {
            (Value::Int(lhs), Value::Int(rhs)) => {
                return Value::Int(lhs + rhs)
            },
            (Value::Float(lhs), Value::Float(rhs)) => {
                return Value::Float(lhs + rhs)
            },
            (lhs, rhs) => {
                err_op_mismatch("subtract", lhs, rhs);
            },
        }
    }

    pub fn sub(lhs: Self, rhs: Self) -> Self {
        match (lhs, rhs) {
            (Value::Int(lhs), Value::Int(rhs)) => {
                return Value::Int(lhs - rhs)
            },
            (Value::Float(lhs), Value::Float(rhs)) => {
                return Value::Float(lhs - rhs)
            },
            (lhs, rhs) => {
                err_op_mismatch("subtract", lhs, rhs);
            },
        }
    }

    pub fn boolean(value: Self) -> bool {
        match value {
            Value::Bool(boolean) => {
                return boolean
            },
            value => {
                err_is_not(value, "boolean")
            },
        }
    }
}

#[derive(Debug, Default)]
pub struct Vars {
    assignments: HashMap<Ident, Value>,
    functions: HashMap<Ident, Ptr<Block>>,
}

impl Vars {
    pub fn new() -> Self {
        Default::default()
    }
}

pub fn eval(expr: &Expr, vars: &mut Vars) -> Value {
    match expr {
        Expr::Block(block) => {
            let mut returns = Value::None;
            for expr in block.iter() {
                returns = eval(expr, vars);
            }
            returns
        },
        Expr::Conditional { condition, then_expr, else_expr } => {
            if Value::boolean(eval(condition, vars)) {
                return eval(&Expr::Block(then_expr.clone()), vars);
            } else {
                return eval(&Expr::Block(else_expr.clone()), vars);
            }
        },
        Expr::Assign { lhs, rhs } => {
            match eval(lhs, vars) {
                Value::Ident(lhs) => {
                    let rhs = eval(rhs, vars);
                    vars.assignments.insert(lhs, rhs);
                    return Value::None;
                },
                lhs => {
                    err_op_mismatch("assign", lhs, eval(rhs, vars));
                }
            }
        },
        Expr::Add { lhs, rhs } => {
            match eval(lhs, vars) {
                Value::Ident(ident) => {
                    match vars.assignments.get(&ident) {
                        Some(value) => {
                            Value::add(value.clone(), eval(rhs, vars))
                        },
                        None => {
                            err_unknown_ident(ident);
                        },
                    }
                },
                _ => {
                    Value::add(eval(lhs, vars), eval(rhs, vars))
                }
            }
        },
        Expr::Sub { lhs, rhs } => {
            match eval(lhs, vars) {
                Value::Ident(ident) => {
                    match vars.assignments.get(&ident) {
                        Some(value) => {
                            Value::sub(value.clone(), eval(rhs, vars))
                        },
                        None => {
                            err_unknown_ident(ident);
                        },
                    }
                },
                _ => {
                    Value::sub(eval(lhs, vars), eval(rhs, vars))
                }
            }
        },
        Expr::FunctionDef { ident, param, body } => {
            let body = Ptr::new(Block((*body).clone()));
            vars.functions.insert(ident.clone(), body);

            for (parameter, default_value) in param {
                vars.assignments.insert(parameter.ident.prefix(ident), default_value.clone());
            }

            return Value::None;
        },
        Expr::FunctionCall { ident, param } => {
            for (parameter, value) in param {
                vars.assignments.insert(parameter.ident.prefix(ident), value.clone());
            }
            
            match vars.functions.get(ident).cloned() {
                Some(function) => {
                    return eval(&Expr::Block(function), vars);
                },
                None => {
                    dbg!(vars);
                    err_unknown_ident(ident.clone());
                },
            }
        },
        Expr::Value(value) => {
            return value.clone()
        },
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Ir {
    Push(Patch),
    Branch(Patch),
    Add,
    Sub,
    Field(Patch),
    Store(Patch),
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Patch {
    Ident(Ident),
    Address(u32),
    Integer(i64),
    Float64(f64),
}

pub struct IrStream {
    code: Vec<Ir>,
    vars: Vars,
}

impl IrStream {
    fn push(&mut self, code: Ir) {
        self.code.push(code);
    }
}

fn emit(expr: &Expr) -> Vec<Ir> {
    let mut code = IrStream {
        code: Vec::new(),
        vars: Vars::new(),
    };

    r_emit(expr, &mut code);

    dbg!(code.code)
}

fn r_emit(expr: &Expr, code: &mut IrStream) {
    match expr {
        Expr::Block(block) => {
            for expr in block.iter() {
                r_emit(expr, code);
            }
        },
        
        Expr::Conditional { condition, then_expr, else_expr } => todo!(),

        Expr::Assign { lhs, rhs } => {
            let lhs = eval(lhs, &mut code.vars);
            if let Value::Ident(ident) = lhs {
                r_emit(rhs, code);
                code.push(Ir::Store(Patch::Ident(ident)));
            } else {
                err_is_not(lhs, "identifier");
            }
        },

        Expr::Add { lhs, rhs } => {
            r_emit(lhs, code);
            r_emit(rhs, code);
            code.push(Ir::Add);
        },

        Expr::Sub { lhs, rhs } => {
            r_emit(lhs, code);
            r_emit(rhs, code);
            code.push(Ir::Sub);
        },

        Expr::FunctionDef { ident, param, body } => todo!(),

        Expr::FunctionCall { ident, param } => todo!(),

        Expr::Value(value) => {
            match value {
                Value::Int(value) => {
                    code.push(Ir::Push(Patch::Integer(*value)))
                },
                Value::Float(value) => {
                    code.push(Ir::Push(Patch::Float64(*value)))
                },
                Value::Bool(_) => todo!(),
                Value::Ident(value) => {
                    code.push(Ir::Field(Patch::Ident(value.clone())))
                },
                Value::None => {
                    /* nop */
                },
            }
        },
    }
}

fn err_op_mismatch(op: &'static str, lhs: Value, rhs: Value) -> ! {
    panic!("operator mismatch, cannot {op} {lhs:?} and {rhs:?}")
}

fn err_is_not(value: Value, kind: &'static str) -> ! {
    panic!("{value:?} is not a {kind}")
}

fn err_unknown_ident(ident: Ident) -> ! {
    panic!("unknown identifier: {ident:?}")
}

// pub fn add(left: usize, right: usize) -> usize {
//     left + right
// }

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[ignore]
    fn interp_add() {
        //let ast = Expr::add(
        //    Expr::value(Value::int(27)),
        //    Expr::value(Value::int(15))
        //);

        let ast = Expr::block([
            Expr::Assign {
                lhs: Ptr::new(Expr::Value(Value::Ident(Ident::from("x")))),
                rhs: Ptr::new(Expr::Add {
                    lhs: Ptr::new(Expr::Value(Value::Int(-99))),
                    rhs: Ptr::new(Expr::Value(Value::Int(101))),
                })
            },
            Expr::Sub {
                lhs: Ptr::new(Expr::Value(Value::Ident(Ident::from("x")))),
                rhs: Ptr::new(Expr::Value(Value::Int(1077))),
            }
        ]);

        let mut vars = Vars::new();
        let result = eval(&ast, &mut vars);
        let emitted = emit(&ast);
        println!("{:?}", result);
    }
}
