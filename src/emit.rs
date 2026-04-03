use crate::ast::{BinOpK, Expr, ExprK, LitK, LoopK, SymTable};
use crate::error::{err_is_not, err_sym_is_not};
use crate::eval::Eval;
use crate::ir::{Ir, IrCode, Marker};
use crate::value::Value;

#[derive(Default)]
pub struct Emit {
    code: IrCode,
    syms: SymTable,
}

impl Emit {
    pub fn emit(expr: &Expr) -> IrCode {
        let emit = Self::default();
        Emit::emit_r(&emit, expr);
        emit.code
    }

    fn emit_r(&self, expr: &Expr) {
        let (_, _) = (expr.astid, expr.span);
        let code = &self.code;
        let _syms = &self.syms;

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
                match Eval::eval(expr) {
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
                    self.emit_r(else_block);

                    let else_jump_offset = self.code.len() as isize - jump_else_location as isize;
                    self.code.patch(Ir::Jump(Marker::Offset(else_jump_offset)), jump_else_location, Ir::Jump(Marker::Temporary));

                    cond_jump_offset += 1;
                }

                self.code.patch(Ir::JumpFalse(Marker::Offset(cond_jump_offset)), cond_location, Ir::JumpFalse(Marker::Temporary));
            },
            ExprK::Call(path, _args) => {
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
}
