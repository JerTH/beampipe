use std::{cell::RefCell, collections::HashMap};

use crate::ast::{
    BinOpK, UnaryOpK, Expr, ExprK, FnArg, LitK, LocalK, LoopK, Path, Ptr, Fn, Span,
};
use crate::error::{RuntimeError, RuntimeErrorK};
use crate::value::Value;

fn path_key(path: &Path) -> u64 {
    if path.list.len() == 1 {
        return path.list[0].name.key();
    }
    const SEPARATOR: u64 = 0x1234_5678_9ABC_DEF0;
    let mut state: u64 = 0xCBF2_9CE4_8422_2325;
    for ident in &path.list {
        state ^= ident.name.key();
        state = state.wrapping_mul(0x0100_0000_01B3);
        state ^= SEPARATOR;
        state = state.wrapping_mul(0x0100_0000_01B3);
    }
    state
}

fn path_to_string(path: &Path) -> String {
    path.list.iter()
        .map(|ident| ident.as_string())
        .collect::<Vec<_>>()
        .join("::")
}

struct Binding {
    key: u64,
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

    fn declare(&mut self, key: u64, value: Value) {
        self.bindings.push(Binding { key, value });
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
    func: RefCell<HashMap<u64, Ptr<Fn>>>,
}

impl Eval {
    pub fn eval(expr: &Expr) -> Result<Value, RuntimeError> {
        let state = Self {
            env: Default::default(),
            func: Default::default(),
        };
        state.eval_r(expr)
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
                self.env.borrow_mut().declare(key, value);
                Ok(Value::None)
            },
            ExprK::Item(_item) => {
                todo!()
            },
            ExprK::Lit(lit) => {
                match &lit.kind {
                    LitK::Int => {
                        let v = lit.symbol.parse::<i64>().map_err(|_| RuntimeError::new(
                            RuntimeErrorK::ParseFailure { kind: "i64", text: lit.symbol.as_string() },
                            span,
                        ))?;
                        Ok(Value::Int(v))
                    },
                    LitK::Float => {
                        let v = lit.symbol.parse::<f64>().map_err(|_| RuntimeError::new(
                            RuntimeErrorK::ParseFailure { kind: "f64", text: lit.symbol.as_string() },
                            span,
                        ))?;
                        Ok(Value::Float(v))
                    },
                    LitK::Bool => {
                        let v = lit.symbol.parse::<bool>().map_err(|_| RuntimeError::new(
                            RuntimeErrorK::ParseFailure { kind: "bool", text: lit.symbol.as_string() },
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
                        let key = path_key(path);
                        match self.env.borrow_mut().assign(key) {
                            Some(slot) => *slot = value,
                            None => return Err(RuntimeError::new(
                                RuntimeErrorK::UndeclaredVariable { name: path_to_string(path) },
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
                let key = path_key(path);
                match self.env.borrow().lookup(key) {
                    Some(value) => Ok(value.clone()),
                    None => Err(RuntimeError::new(
                        RuntimeErrorK::UndeclaredVariable { name: path_to_string(path) },
                        span,
                    )),
                }
            },
            ExprK::Fn(func) => {
                let path = func.path.clone();
                let key = path_key(&path);
                {
                    use std::collections::hash_map::Entry;
                    match self.func.borrow_mut().entry(key) {
                        Entry::Occupied(_) => {
                            return Err(RuntimeError::new(
                                RuntimeErrorK::MultipleDeclarations { name: path_to_string(&path) },
                                span,
                            ));
                        }
                        Entry::Vacant(e) => { e.insert(func.clone()); }
                    }
                }

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
        let key = path_key(func);

        let func_def = self.func.borrow().get(&key).cloned();
        if let Some(func_def) = func_def {
            let watermark = self.env.borrow_mut().push_scope();
            for (param, arg) in func_def.sig.params.iter().zip(args.iter()) {
                let resolved = self.eval_r(&arg.expr)?;
                let key = param.ident.name.key();
                self.env.borrow_mut().declare(key, resolved);
            }
            let result = self.eval_r(&func_def.body)?;
            self.env.borrow_mut().pop_scope(watermark);
            Ok(result)
        } else {
            Err(RuntimeError::new(
                RuntimeErrorK::UndeclaredFunction { name: path_to_string(func) },
                call_span,
            ))
        }
    }
}

use std::ops::{Add, Div, Mul, Sub};
