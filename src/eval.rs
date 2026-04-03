use std::{cell::RefCell, collections::HashMap, ops::{Add, Div, Mul, Sub}};

use crate::ast::{
    BinOpK, Expr, ExprK, FnArg, Ident, LitK, LoopK, Path, Ptr, Span, SymTable, Fn,
};
use crate::error::err_is_not;
use crate::value::Value;

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
