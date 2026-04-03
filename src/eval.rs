use std::{cell::RefCell, collections::HashMap, ops::{Add, Div, Mul, Sub}};

use crate::ast::{
    BinOpK, Expr, ExprK, FnArg, LitK, LocalK, LoopK, Path, Ptr, SymTable, Fn,
};
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

pub struct Eval {
    env: RefCell<Env>,
    func: RefCell<HashMap<String, Ptr<Fn>>>,
    symt: RefCell<SymTable>,
}

impl Eval {
    pub fn eval(expr: &Expr) -> Value {
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

    fn eval_r(&self, expr: &Expr) -> Value {
        let (_, _) = (expr.astid, expr.span);

        match &expr.kind {
            ExprK::Empty => {
                Value::None
            },
            ExprK::Semi(expr) => {
                self.eval_r(expr);
                Value::None
            },
            ExprK::Local(local) => {
                let value = match &local.kind {
                    LocalK::Decl => Value::None,
                    LocalK::Init(expr) => self.eval_r(expr),
                };
                let key = local.ident.name.key();
                let name = local.ident.as_string();
                self.env.borrow_mut().declare(key, name, value);
                Value::None
            },
            ExprK::Item(_item) => {
                todo!()
            },
            ExprK::Lit(lit) => {
                match &lit.kind {
                    LitK::Int => Value::Int(lit.symbol.parse::<i64>().expect("expected i64 to parse")),
                    LitK::Float => Value::Float(lit.symbol.parse::<f64>().expect("expected f64 to parse")),
                    LitK::Bool => Value::Bool(lit.symbol.parse::<bool>().expect("expected boolean to parse")),
                }
            },
            ExprK::Block(block) => {
                let watermark = self.env.borrow_mut().push_scope();
                let mut returns = Value::None;
                for expr in &block.list {
                    returns = self.eval_r(expr);
                }
                self.env.borrow_mut().pop_scope(watermark);
                returns
            },
            ExprK::Assign(lhs, rhs) => {
                let value = self.eval_r(rhs);
                match &lhs.kind {
                    ExprK::Path(path) => {
                        let (key, full_name) = self.resolve_path(path);
                        match self.env.borrow_mut().assign(key) {
                            Some(slot) => *slot = value,
                            None => panic!("assignment to undeclared variable `{}`", full_name),
                        }
                    },
                    _ => {
                        panic!("invalid assignment target");
                    },
                }
                Value::None
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
                        self.eval_r(lhs).lt(self.eval_r(rhs))
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
                let (key, full_name) = self.resolve_path(path);
                match self.env.borrow().lookup(key) {
                    Some(value) => value.clone(),
                    None => panic!("undeclared variable `{}`", full_name),
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
                        self.eval_r(then)
                    },
                    Value::Bool(false) => {
                        if let Some(expr) = else_then {
                            self.eval_r(expr)
                        } else {
                            Value::None
                        }
                    },
                    condr => {
                        panic!("{:?} is not a boolean", condr);
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
                    LoopK::For(_head, _body) => {
                        todo!()
                    },
                    LoopK::While(_cond, _body) => {
                        todo!()
                    },
                    LoopK::Loop(_label, _body) => {
                        todo!()
                    },
                }
            },
        }
    }

    fn call_fn(&self, func: &Ptr<Path>, args: &[Ptr<FnArg>]) -> Value {
        let func_path_string = String::from(func);

        let func_def = self.func.borrow().get(&func_path_string).cloned();
        if let Some(func_def) = func_def {
            let watermark = self.env.borrow_mut().push_scope();
            for (param, arg) in func_def.sig.params.iter().zip(args.iter()) {
                let resolved = self.eval_r(&arg.expr);
                let key = param.ident.name.key();
                let name = param.ident.as_string();
                self.env.borrow_mut().declare(key, name, resolved);
            }
            let result = self.eval_r(&func_def.body);
            self.env.borrow_mut().pop_scope(watermark);
            result
        } else {
            let funcs_clone = self.func.borrow().clone();
            let funcs = funcs_clone.iter().map(|f| f.0).collect::<Vec<_>>();
            panic!("undeclared fn\n{:#?}\n{:#?}\n", func, funcs);
        }
    }
}
