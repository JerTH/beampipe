use crate::ast::{BinOpK, UnaryOpK, Expr, ExprK, Ident, LitK, LocalK, LoopK, Path, Ptr, SymTable};
use crate::error::{RuntimeError, RuntimeErrorK};
use crate::ir::{Ir, IrCode, Marker};

#[derive(Default)]
pub struct Emit {
    code: IrCode,
    syms: SymTable,
}

impl Emit {
    pub fn emit(expr: &Expr) -> Result<IrCode, RuntimeError> {
        let emit = Self::default();
        Emit::emit_r(&emit, expr)?;
        Ok(emit.code)
    }

    fn path_to_ident(&self, path: &Ptr<Path>) -> Ident {
        let name = String::from(path);
        Ident {
            name: self.syms.make(name),
            span: path.span,
        }
    }

    fn emit_r(&self, expr: &Expr) -> Result<(), RuntimeError> {
        let code = &self.code;
        let span = expr.span;

        match &expr.kind {
            ExprK::Empty => {
                code.emit(Ir::Nop);
            },

            ExprK::Semi(expr) => {
                self.emit_r(expr)?;
            },

            ExprK::Local(local) => {
                match &local.kind {
                    LocalK::Init(expr) => {
                        self.emit_r(expr)?;
                    }
                    LocalK::Decl => {
                        code.emit(Ir::Nop);
                    }
                }
                let ident = Ident {
                    name: self.syms.make(local.ident.as_string()),
                    span: local.span,
                };
                code.emit(Ir::Decl(Marker::Ident(ident)));
            },
            ExprK::Item(_) => todo!(),
            ExprK::Lit(lit) => {
                match lit.kind {
                    LitK::Bool => {
                        let v = lit.symbol.parse::<bool>().map_err(|_| RuntimeError::new(
                            RuntimeErrorK::ParseFailure { kind: "boolean", text: lit.symbol.as_string() },
                            span,
                        ))?;
                        code.emit(Ir::Bool(v));
                    },
                    LitK::Int => {
                        let v = lit.symbol.parse::<i64>().map_err(|_| RuntimeError::new(
                            RuntimeErrorK::ParseFailure { kind: "integer", text: lit.symbol.as_string() },
                            span,
                        ))?;
                        code.emit(Ir::Integer(v));
                    },
                    LitK::Float => {
                        let v = lit.symbol.parse::<f64>().map_err(|_| RuntimeError::new(
                            RuntimeErrorK::ParseFailure { kind: "float", text: lit.symbol.as_string() },
                            span,
                        ))?;
                        code.emit(Ir::Float(v));
                    },
                }
            },
            ExprK::Block(block) => {
                code.emit(Ir::ScopeEnter);
                for expr in &block.list {
                    self.emit_r(expr)?;
                }
                code.emit(Ir::ScopeExit);
            },
            ExprK::Assign(lhs, rhs) => {
                match &lhs.kind {
                    ExprK::Path(path) => {
                        self.emit_r(rhs)?;
                        let ident = self.path_to_ident(path);
                        code.emit(Ir::Store(Marker::Ident(ident)));
                    },
                    _ => return Err(RuntimeError::new(
                        RuntimeErrorK::InvalidAssignmentLhs,
                        lhs.span,
                    )),
                }
            },
            ExprK::BinOp(op, lhs, rhs) => {
                self.emit_r(lhs)?;
                self.emit_r(rhs)?;

                match op.kind {
                    BinOpK::Add => { code.emit(Ir::Add) },
                    BinOpK::Sub => { code.emit(Ir::Sub) },
                    BinOpK::Div => { code.emit(Ir::Div) },
                    BinOpK::Mul => { code.emit(Ir::Mul) },
                    BinOpK::CmpLess => { code.emit(Ir::CmpLess) },
                    BinOpK::CmpGreater => { code.emit(Ir::CmpGreater) },
                    BinOpK::Eq => { code.emit(Ir::Eq) },
                    BinOpK::Neq => { code.emit(Ir::Neq) },
                    BinOpK::And => { code.emit(Ir::And) },
                    BinOpK::Or => { code.emit(Ir::Or) },
                }
            },
            ExprK::AssignOp(_, _, _) => {
                todo!()
            },
            ExprK::Path(path) => {
                let ident = self.path_to_ident(path);
                code.emit(Ir::Load(Marker::Ident(ident)));
            },
            // A function declaration
            ExprK::Fn(func) => {
                let path = &func.path;
                let symbol = self.syms.make(path);
                self.code.emit(Ir::Symbol(symbol));

                if let ExprK::Block(body) = &func.body.kind {
                    for expr in &body.list {
                        self.emit_r(expr)?;
                    }
                } else {
                    return Err(RuntimeError::new(
                        RuntimeErrorK::FnBodyNotBlock,
                        func.body.span,
                    ));
                }

                self.code.emit(Ir::Return);
            },

            ExprK::If(pred, blk, else_blk) => {
                self.emit_r(pred)?;

                let cond_location = self.code.len();
                self.code.emit(Ir::JumpFalse(Marker::Temporary));

                self.emit_r(blk)?;

                let mut cond_jump_offset = self.code.len() as isize - cond_location as isize;

                if let Some(else_block) = else_blk {
                    let jump_else_location = self.code.len();
                    self.code.emit(Ir::Jump(Marker::Temporary));
                    self.emit_r(else_block)?;

                    let else_jump_offset = self.code.len() as isize - jump_else_location as isize;
                    self.code.patch(Ir::Jump(Marker::Offset(else_jump_offset)), jump_else_location, Ir::Jump(Marker::Temporary), span)?;

                    cond_jump_offset += 1;
                }

                self.code.patch(Ir::JumpFalse(Marker::Offset(cond_jump_offset)), cond_location, Ir::JumpFalse(Marker::Temporary), span)?;
            },
            ExprK::Call(path, _args) => {
                match &path.kind {
                    ExprK::Path(fnpath) => {
                        let fn_path_string = String::from(fnpath);
                        let sym = self.syms.make(fn_path_string);
                        self.code.emit(Ir::Call(sym));
                    },
                    _ => {
                        return Err(RuntimeError::new(
                            RuntimeErrorK::CallTargetNotPath,
                            path.span,
                        ));
                    }
                }
            },
            ExprK::UnaryOp(kind, operand) => {
                self.emit_r(operand)?;
                match kind {
                    UnaryOpK::Neg => { code.emit(Ir::Neg) },
                    UnaryOpK::Not => { code.emit(Ir::Not) },
                }
            },
            ExprK::Loop(loop_block) => {
                match &loop_block.kind {
                    LoopK::For(_head, _body) => {
                        todo!()
                    },
                    LoopK::While(cond, body) => {
                        let loop_head = self.code.len();
                        self.emit_r(cond)?;

                        let cond_location = self.code.len();
                        self.code.emit(Ir::JumpFalse(Marker::Temporary));

                        self.code.emit(Ir::ScopeEnter);
                        for expr in &body.list {
                            self.emit_r(expr)?;
                        }
                        self.code.emit(Ir::ScopeExit);

                        let jump_back_offset = loop_head as isize - self.code.len() as isize;
                        self.code.emit(Ir::Jump(Marker::Offset(jump_back_offset)));

                        let exit_offset = self.code.len() as isize - cond_location as isize;
                        self.code.patch(
                            Ir::JumpFalse(Marker::Offset(exit_offset)),
                            cond_location,
                            Ir::JumpFalse(Marker::Temporary),
                            span,
                        )?;
                    },
                    LoopK::Loop(_label, _body) => {
                        todo!()
                    },
                }
            },
        }
        Ok(())
    }
}
