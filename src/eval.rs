use std::{cell::RefCell, collections::HashMap};

use crate::ast::{
    BinOpK, UnaryOpK, Expr, ExprK, FnArg, LitK, LocalK, LoopK, Path, Ptr, SymTable, Fn, Span,
};
use crate::error::{RuntimeError, RuntimeErrorK};
use crate::value::Value;

struct Binding {
    key: u64,
    name: String,
    value: Value,
}

struct Env {
    bindings: Vec<Binding>,
}

impl Default for Env {
    fn default() -> Self {
        Env {
            bindings: Vec::new(),
        }
    }
}

impl Env {
    fn push_scope(&mut self) -> usize {
        self.bindings.len()
    }

    fn pop_scope(&mut self, watermark: usize) {
        self.bindings.truncate(watermark);
    }

    fn declare(&mut self, key: u64, name: String, value: Value) {
        self.bindings.push(Binding { key, name, value });
    }

    fn lookup(&self, key: u64) -> Option<&Value> {
        self.bindings.iter().rev().find(|b| b.key == key).map(|b| &b.value)
    }

    fn assign(&mut self, key: u64) -> Option<&mut Value> {
        self.bindings.iter_mut().rev().find(|b| b.key == key).map(|b| &mut b.value)
    }
}

fn with_span(result: Result<Value, RuntimeErrorK>, span: Span) -> Result<Value, RuntimeError> {
    result.map_err(|kind| RuntimeError::new(kind, span))
}

pub struct Eval {
    env: RefCell<Env>,
    func: RefCell<HashMap<String, Ptr<Fn>>>,
    symt: RefCell<SymTable>,
}

impl Eval {
    pub fn eval(expr: &Expr) -> Result<Value, RuntimeError> {
        let state = Self {
            env: Default::default(),
            symt: Default::default(),
            func: Default::default(),
        };
        state.eval_r(expr)
    }

    fn resolve_path(&self, path: &Ptr<Path>) -> (u64, String) {
        let full_name: String = path.list.iter().fold(String::new(), |acc, x| {
            format!("{}::{}", acc, x)
        })
        .trim_start_matches("::")
        .into();

        let key = self.symt.borrow().make(full_name.clone()).key();
        (key, full_name)
    }

    fn eval_r(&self, expr: &Expr) -> Result<Value, RuntimeError> {
        let span = expr.span;

        match &expr.kind {
            ExprK::Empty => {
                Ok(Value::None)
            },
            ExprK::Semi(expr) => {
                self.eval_r(expr)?;
                Ok(Value::None)
            },
            ExprK::Local(local) => {
                let value = match &local.kind {
                    LocalK::Decl => Value::None,
                    LocalK::Init(expr) => self.eval_r(expr)?,
                };
                let key = local.ident.name.key();
                let name = local.ident.as_string();
                self.env.borrow_mut().declare(key, name, value);
                Ok(Value::None)
            },
            ExprK::Item(_item) => {
                todo!()
            },
            ExprK::Lit(lit) => {
                match &lit.kind {
                    LitK::Int => {
                        let v = lit.symbol.parse::<i64>().map_err(|_| RuntimeError::new(
                            RuntimeErrorK::LiteralParseFailure { kind: "i64", text: lit.symbol.as_string() },
                            span,
                        ))?;
                        Ok(Value::Int(v))
                    },
                    LitK::Float => {
                        let v = lit.symbol.parse::<f64>().map_err(|_| RuntimeError::new(
                            RuntimeErrorK::LiteralParseFailure { kind: "f64", text: lit.symbol.as_string() },
                            span,
                        ))?;
                        Ok(Value::Float(v))
                    },
                    LitK::Bool => {
                        let v = lit.symbol.parse::<bool>().map_err(|_| RuntimeError::new(
                            RuntimeErrorK::LiteralParseFailure { kind: "bool", text: lit.symbol.as_string() },
                            span,
                        ))?;
                        Ok(Value::Bool(v))
                    },
                }
            },
            ExprK::Block(block) => {
                let watermark = self.env.borrow_mut().push_scope();
                let mut returns = Value::None;
                for expr in &block.list {
                    returns = self.eval_r(expr)?;
                }
                self.env.borrow_mut().pop_scope(watermark);
                Ok(returns)
            },
            ExprK::Assign(lhs, rhs) => {
                let value = self.eval_r(rhs)?;
                match &lhs.kind {
                    ExprK::Path(path) => {
                        let (key, full_name) = self.resolve_path(path);
                        match self.env.borrow_mut().assign(key) {
                            Some(slot) => *slot = value,
                            None => return Err(RuntimeError::new(
                                RuntimeErrorK::UndeclaredVariable { name: full_name },
                                lhs.span,
                            )),
                        }
                    },
                    _ => {
                        return Err(RuntimeError::new(
                            RuntimeErrorK::InvalidAssignmentTarget,
                            lhs.span,
                        ));
                    },
                }
                Ok(Value::None)
            },
            ExprK::BinOp(op, lhs, rhs) => {
                let lval = self.eval_r(lhs)?;
                let rval = self.eval_r(rhs)?;
                with_span(match op.kind {
                    BinOpK::Add => lval.add(rval),
                    BinOpK::Sub => lval.sub(rval),
                    BinOpK::Div => lval.div(rval),
                    BinOpK::Mul => lval.mul(rval),
                    BinOpK::CmpLess => lval.lt(rval),
                    BinOpK::CmpGreater => lval.gt(rval),
                    BinOpK::Eq => lval.eq_val(rval),
                    BinOpK::Neq => lval.neq(rval),
                    BinOpK::And => lval.and(rval),
                    BinOpK::Or => lval.or(rval),
                }, span)
            },
            ExprK::AssignOp(_, _, _) => {
                todo!()
            },
            ExprK::Path(path) => {
                let (key, full_name) = self.resolve_path(path);
                match self.env.borrow().lookup(key) {
                    Some(value) => Ok(value.clone()),
                    None => Err(RuntimeError::new(
                        RuntimeErrorK::UndeclaredVariable { name: full_name },
                        span,
                    )),
                }
            },
            ExprK::Fn(func) => {
                let path = func.path.clone();
                let path_str = String::from(&path);
                if self.func.borrow().contains_key(&path_str) {
                    return Err(RuntimeError::new(
                        RuntimeErrorK::MultipleDeclarations { name: path_str },
                        span,
                    ));
                }
                self.func.borrow_mut().insert(path_str, func.clone());

                // main entry-point for eval
                if let Some(fn_path_name) = path.list.last() {
                    if fn_path_name.as_string().as_str() == "main" {
                        let path_ptr = Ptr::new(path);

                        let main_args = Vec::new();
                        return self.call_fn(&path_ptr, &main_args, span);
                    }
                }

                Ok(Value::None)
            },
            ExprK::If(cond, then, else_then) => {
                let condr = self.eval_r(cond)?;
                match condr {
                    Value::Bool(true) => {
                        self.eval_r(then)
                    },
                    Value::Bool(false) => {
                        if let Some(expr) = else_then {
                            self.eval_r(expr)
                        } else {
                            Ok(Value::None)
                        }
                    },
                    _ => {
                        Err(RuntimeError::new(
                            RuntimeErrorK::NonBooleanCondition { found: condr.type_name() },
                            cond.span,
                        ))
                    }
                }
            },
            ExprK::Call(name, args) => {
                match name.kind {
                    ExprK::Path(ref fnpath) => {
                        self.call_fn(fnpath, args, span)
                    }
                    _ => {
                        Err(RuntimeError::new(
                            RuntimeErrorK::InvalidCallTarget,
                            name.span,
                        ))
                    }
                }
            },
            ExprK::UnaryOp(kind, operand) => {
                let val = self.eval_r(operand)?;
                with_span(match kind {
                    UnaryOpK::Neg => val.neg(),
                    UnaryOpK::Not => val.not(),
                }, span)
            },
            ExprK::Loop(loop_block) => {
                match &loop_block.kind {
                    LoopK::For(_head, _body) => {
                        todo!()
                    },
                    LoopK::While(cond, body) => {
                        while self.eval_r(cond)? == Value::Bool(true) {
                            let watermark = self.env.borrow_mut().push_scope();
                            for expr in &body.list {
                                self.eval_r(expr)?;
                            }
                            self.env.borrow_mut().pop_scope(watermark);
                        }
                        Ok(Value::None)
                    },
                    LoopK::Loop(_label, _body) => {
                        todo!()
                    },
                }
            },
        }
    }

    fn call_fn(&self, func: &Ptr<Path>, args: &[Ptr<FnArg>], call_span: Span) -> Result<Value, RuntimeError> {
        let func_path_string = String::from(func);

        let func_def = self.func.borrow().get(&func_path_string).cloned();
        if let Some(func_def) = func_def {
            let watermark = self.env.borrow_mut().push_scope();
            for (param, arg) in func_def.sig.params.iter().zip(args.iter()) {
                let resolved = self.eval_r(&arg.expr)?;
                let key = param.ident.name.key();
                let name = param.ident.as_string();
                self.env.borrow_mut().declare(key, name, resolved);
            }
            let result = self.eval_r(&func_def.body)?;
            self.env.borrow_mut().pop_scope(watermark);
            Ok(result)
        } else {
            Err(RuntimeError::new(
                RuntimeErrorK::UndeclaredFunction { name: func_path_string },
                call_span,
            ))
        }
    }
}

// Bring std ops into scope for the BinOp match
use std::ops::{Add, Div, Mul, Sub};
