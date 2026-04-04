use std::{fmt::Display, cell::Cell};

use crate::{token::{Token, TokenK}, ast::{AstId, Fn, Expr, ExprK, Span, Ptr, Sym, TyK, Ty, FnSig, FnParam, Blk, Ident, Lit, LitK, BinOp, BinOpK, UnaryOpK, Path, SymTable, FnArg, Local, LocalK, Loop, LoopK}};
use crate::error::{ParserError, ParserErrorK};



#[derive(Debug, Default)]
pub struct Parse {
    source: String,
    tokens: Vec<Token>,
    astout: Option<Expr>,
}

impl Display for Parse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Parse Results:\n\n")?;
        write!(f, "Source\n======\n{}\n======\n\n", self.source)?;
        write!(f, "Tokens\n======\n")?;
        for token in &self.tokens {
            let slice = &self.source[token.span.bgn..=token.span.end];
            writeln!(f, "{:11}({:?})\t\"{}\"", token.kind, token.span, slice)?;
        }
        write!(f, "======\n\nAstOut\n======\n")?;
        write!(f, "{:?}\n======\n", self.astout)?;
        Ok(())
    }
}

impl Parse {
    pub fn parse<S: Into<String>>(source: S) -> Result<Expr, ParserError> {
        let source: String = source.into();
        let tokens = crate::lex::tokenize(&source);
        let astout: Option<Expr> = None;

        let mut parse = Self {
            source,
            tokens,
            astout,
        };

        parse.build_tree()
    }

    pub fn build_tree(&mut self) -> Result<Expr, ParserError> {
        let parser = Parser {
            source: self.source.clone(),
            tokens: self.tokens.clone(),
            cursor: Cell::new(0),
            symtab: SymTable::new(),
        };

        parser.parse()
    }

}

trait BindingExt<L, R> {
    fn left(&self) -> L;
    fn right(&self) -> R;
}

impl<L, R> BindingExt<L, R> for (L, R)
where
    L: Clone,
    R: Clone
{
    fn left(&self) -> L {
        self.0.clone()
    }

    fn right(&self) -> R {
        self.1.clone()
    }
}




#[derive(Debug, Clone)]
pub struct Parser {
    source: String,
    tokens: Vec<Token>,
    cursor: Cell<usize>,
    symtab: SymTable,
}

impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl Parser {
    pub fn new() -> Self {
        Self {
            source: String::new(),
            cursor: Cell::new(0),
            tokens: Vec::new(),
            symtab: SymTable::new(),
        }
    }

    fn token(&self) -> Token {
        match self.tokens.get(self.cursor.get()).cloned() {
            Some(token) => token,
            None => {
                let last = self.tokens.len();
                Token {
                    kind: TokenK::Eof,
                    span: Span { bgn: last, end: last },
                }
            },
        }
    }

    pub fn peek(&self, offset: isize) -> Option<Token> {
        let offset_cursor = self.cursor.get().checked_add_signed(offset)?;
        self.tokens.get(offset_cursor).cloned()
    }

    pub fn peek_kind(&self, offset: isize) -> Option<TokenK> {
        let offset_cursor = self.cursor.get().checked_add_signed(offset)?;
        self.tokens.get(offset_cursor).cloned().map(|t| t.kind)
    }

    fn advance(&self) -> Token {
        self.cursor.set(self.cursor.get() + 1);
        self.token()
    }

    /// Checks that the kind of the current token is equal to `tokenk`
    /// and advances the token pointer if it is
    fn eat(&self, tokenk: TokenK) -> Result<Token, ParserError> {
        let current = self.token();
        if current.kind != tokenk {
            return Err(ParserError::single(ParserErrorK::UnexpectedToken {
                expected: tokenk,
                found: current.kind,
                span: current.span,
            }));
        }
        Ok(self.advance())
    }

    /// Parse a token stream
    fn parse(&self) -> Result<Expr, ParserError> {
        let mut expressions = Vec::new();

        loop {
            match self.parse_expression(0) {
                Ok(Some(expr)) => expressions.push(Ptr::new(expr)),
                Ok(None) => break,
                Err(e) => return Err(e),
            }
        }

        let blk = Blk::new(expressions, Span::none());
        let block = ExprK::Block(Ptr::new(blk));
        Ok(Expr::new(Span::none(), block))
    }

    /// Parses an expression using Pratt's method
    ///
    /// Parsing begins with the token pointed to by self.token()
    fn parse_expression(&self, right_binding: usize) -> Result<Option<Expr>, ParserError> {
        // TokenK::Eof is handled in parse_null_denotation

        let mut left = match self.parse_null_denotation(self.token())? {
            Some(expr) => expr,
            None => return Ok(None),
        };

        // as long as the right associative binding power is less than the left associative
        // binding power of the following token, we process successive infix and suffix
        // tokens with the left denotation method. If the token is not left associative,
        // we return the null associative expression already stored in `left`
        while right_binding < self.token().infix_binding().left() {
            left = match self.parse_left_denotation(left.clone(), self.token()) {
                Ok(Some(expr)) => expr,
                Ok(None) => break,
                Err(e) => return Err(e),
            };
        }

        Ok(Some(left))
    }

    /// Parses a qualified or unqualified path starting with the current token
    fn parse_path(&self) -> Result<Option<Path>, ParserError> {
        // todo: may need special handling for primitive types
        // type parsing is sort of handled here but might be
        // better in its own separate solution

        let mut list = Vec::new();
        while (self.token().kind == TokenK::LitIdent)
            | (self.token().kind == TokenK::Ty)
        {
            let ident = match self.parse_ident()? {
                Some(ident) => ident,
                None => break,
            };
            list.push(ident);

            if let Some(TokenK::ColCol) = self.peek_kind(1) {
                self.eat(TokenK::ColCol)?;
                self.advance();
            } else {
                break;
            }
        }

        if list.is_empty() {
            return Ok(None);
        }

        let span = Span::new(
            list.first().unwrap().span.bgn,
            list.last().unwrap().span.end,
        );

        Ok(Some(Path::new(list, span)))
    }

    fn require_path(&self, context: &'static str) -> Result<Path, ParserError> {
        let span = self.token().span;
        self.parse_path()?.ok_or_else(|| {
            ParserError::single(ParserErrorK::ExpectedPath { context, span })
        })
    }

    fn parse_binop(&self, left: Expr, token: Token) -> Result<Option<Expr>, ParserError> {
        let kind = match token.kind {
            TokenK::OpAdd => BinOpK::Add,
            TokenK::OpSub => BinOpK::Sub,
            TokenK::OpMul => BinOpK::Mul,
            TokenK::OpDiv => BinOpK::Div,
            TokenK::OpLess => BinOpK::CmpLess,
            TokenK::OpGreater => BinOpK::CmpGreater,
            TokenK::EqEq => BinOpK::Eq,
            TokenK::BangEq => BinOpK::Neq,
            TokenK::AmpAmp => BinOpK::And,
            TokenK::PipePipe => BinOpK::Or,
            _ => {
                return Err(ParserError::single(ParserErrorK::InvalidBinOp {
                    span: token.span,
                }));
            }
        };

        let binding = self.token().infix_binding().left();

        let lhs = left;
        self.advance();
        let rhs = match self.parse_expression(binding)? {
            Some(expr) => expr,
            None => return Ok(None),
        };

        let opspan = token.span;
        let exspan = Span::new(lhs.span.bgn, rhs.span.end);
        let binop = BinOp { span: opspan, kind };
        let exprk = ExprK::BinOp(binop, Ptr::new(lhs), Ptr::new(rhs));
        Ok(Some(Expr::new(exspan, exprk)))
    }

    fn parse_infix_op(&self, left: Expr, token: Token) -> Result<Option<Expr>, ParserError> {
        match token.kind {
            TokenK::ColCol    |
            TokenK::OpGreater |
            TokenK::OpLess    |
            TokenK::OpAdd     |
            TokenK::OpSub     |
            TokenK::OpMul     |
            TokenK::OpDiv     |
            TokenK::EqEq      |
            TokenK::BangEq    |
            TokenK::AmpAmp    |
            TokenK::PipePipe => {
                self.parse_binop(left, token)
            },
            _ => {
                Err(ParserError::single(ParserErrorK::InvalidInfixOp {
                    span: token.span,
                }))
            }
        }
    }

    fn parse_assignment(&self, lhs: Expr, rhs: Expr) -> Result<Option<Expr>, ParserError> {
        match lhs.kind {
            ExprK::Path(path) => {
                let exspan = Span::new(lhs.span.bgn, rhs.span.end);
                let path = Expr::new(lhs.span, ExprK::Path(path));
                let exprk = ExprK::Assign(Ptr::new(path), Ptr::new(rhs));
                Ok(Some(Expr::new(exspan, exprk)))
            },
            _ => {
                Err(ParserError::single(ParserErrorK::InvalidLValue {
                    span: lhs.span,
                }))
            }
        }
    }

    fn parse_reverse_infix_op(&self, left: Expr, token: Token) -> Result<Option<Expr>, ParserError> {
        let binding = token.infix_binding().right();
        self.advance();
        let lhs = left;
        let rhs = match self.parse_expression(binding)? {
            Some(expr) => expr,
            None => return Ok(None),
        };

        match token.kind {
            TokenK::OpEq => {
                self.parse_assignment(lhs, rhs)
            },
            _ => {
                Err(ParserError::single(ParserErrorK::InvalidInfixOp {
                    span: token.span,
                }))
            }
        }
    }

    fn parse_left_denotation(&self, left: Expr, token: Token) -> Result<Option<Expr>, ParserError> {
        let kind = token.kind;

        match kind {
            TokenK::ColCol    |
            TokenK::OpGreater |
            TokenK::OpLess    |
            TokenK::OpAdd     |
            TokenK::OpSub     |
            TokenK::OpMul     |
            TokenK::OpDiv     |
            TokenK::EqEq      |
            TokenK::BangEq    |
            TokenK::AmpAmp    |
            TokenK::PipePipe => {
                self.parse_infix_op(left, token)
            },
            TokenK::OpEq => {
                self.parse_reverse_infix_op(left, token)
            },
            TokenK::Semi => {
                let span = left.span;
                let exprk = ExprK::Semi(Ptr::new(left));
                Ok(Some(Expr::new(span, exprk)))
            },
            _ => {
                Err(ParserError::single(ParserErrorK::UnexpectedToken {
                    expected: TokenK::Eof,
                    found: kind,
                    span: token.span,
                }))
            }
        }
    }


    /// Parses the current token as a literal, returning it
    ///
    /// Leaves the cursor directly after the literal
    fn parse_literal(&self) -> Result<Option<Expr>, ParserError> {
        let token = self.token();

        let kind = token.kind;
        let span = token.span;

        let kind = match kind {
            TokenK::LitInt => LitK::Int,
            TokenK::LitFloat => LitK::Float,
            TokenK::LitBool => LitK::Bool,
            _ => return Ok(None),
        };

        let symbol = self.make_symbol(&token);
        let literal = Lit { symbol, kind };

        let exprk = ExprK::Lit(Ptr::new(literal));

        self.advance();
        Ok(Some(Expr::new(span, exprk)))
    }

    /// Parses the current token as an ident, returning it
    ///
    /// Leaves the cursor directly after the ident
    fn parse_ident(&self) -> Result<Option<Ident>, ParserError> {
        let token = self.token();
        if token.kind != TokenK::LitIdent && token.kind != TokenK::Ty {
            return Ok(None);
        }
        let span = token.span;
        let sym = self.make_symbol(&token);
        let ident = Ident { name: sym, span };

        self.advance();
        Ok(Some(ident))
    }

    fn parse_prefix_op(&self, token: Token) -> Result<Option<Expr>, ParserError> {
        let kind = match token.kind {
            TokenK::OpSub => UnaryOpK::Neg,
            TokenK::OpBang => UnaryOpK::Not,
            _ => {
                return Err(ParserError::single(ParserErrorK::InvalidPrefixOp {
                    span: token.span,
                }));
            }
        };

        let binding = token.prefix_binding().1;
        self.advance();
        let operand = match self.parse_expression(binding)? {
            Some(expr) => expr,
            None => return Ok(None),
        };

        let span = Span::new(token.span.bgn, operand.span.end);
        let exprk = ExprK::UnaryOp(kind, Ptr::new(operand));
        Ok(Some(Expr::new(span, exprk)))
    }

    fn parse_block(&self) -> Result<Option<Blk>, ParserError> {
        let mut span = self.token().span;

        self.eat(TokenK::LBrace)?;

        let mut expressions = Vec::new();

        while let Some(mut expr) = self.parse_expression(0)? {

            // Is it a semi-colon postfix expression?
            if self.peek_kind(0) == Some(TokenK::Semi) {
                let span = expr.span;
                expr = Expr {
                    astid: expr.astid,
                    kind: ExprK::Semi(Ptr::new(expr)),
                    span: Span::new(span.bgn, self.token().span.end),
                };

                self.eat(TokenK::Semi)?;
            }

            expressions.push(Ptr::new(expr));
        }

        self.eat(TokenK::RBrace)?;

        if expressions.is_empty() {
            span = Span::new(span.bgn, self.token().span.end);
        } else {
            span = Span::new(
                expressions.first().unwrap().span.bgn,
                expressions.last().unwrap().span.end
            );
        }

        Ok(Some(Blk::new(expressions, span)))
    }

    fn parse_function_args(&self) -> Result<Vec<FnArg>, ParserError> {
        self.eat(TokenK::LParen)?;
        let mut args = Vec::new();

        while self.token().kind != TokenK::RParen {
            match self.token().kind {
                TokenK::Comma => {
                    self.advance();
                    continue;
                },
                _ => {
                    let expr = match self.parse_expression(0)? {
                        Some(expr) => expr,
                        None => break,
                    };
                    let span = expr.span;
                    args.push(FnArg::new(expr, span))
                }
            }
        }
        self.eat(TokenK::RParen)?;

        Ok(args)
    }

    /// Parses a functions parameters enclosed in parens.
    ///
    /// After parsing the token cursor will be pointing to just after the right paren
    fn parse_function_params(&self) -> Result<Vec<FnParam>, ParserError> {
        self.eat(TokenK::LParen)?;
        let mut params = Vec::new();

        while self.token().kind != TokenK::RParen {
            match self.token().kind {
                TokenK::Comma => {
                    self.advance();
                    continue;
                }
                TokenK::LitIdent => {
                    // the name of the parameter
                    let ident = self.parse_ident()?.expect("ident check preceded this");
                    self.eat(TokenK::Colon)?;

                    // the type of the parameter
                    let typath = self.require_path("parameter type")?;
                    let tyspan = typath.span;
                    let tyk = TyK::Path(Ptr::new(typath));
                    let ty = Ty::new(tyk, tyspan);

                    let span = Span::new(ident.span.bgn, ty.span.end);
                    let param = FnParam::new(ty, ident, span);

                    params.push(param);
                },
                _ => {
                    let token = self.token();
                    return Err(ParserError::single(ParserErrorK::UnexpectedToken {
                        expected: TokenK::LitIdent,
                        found: token.kind,
                        span: token.span,
                    }));
                }
            }
        }
        self.eat(TokenK::RParen)?;
        Ok(params)
    }

    /// Parses a function signature
    fn parse_function_sig(&self) -> Result<FnSig, ParserError> {
        let mut span = self.token().span;
        let params = self.parse_function_params()?;

        // Has a return type?
        let mut ret = None;
        if let Some(TokenK::Arrow) = self.peek_kind(0) {
            self.eat(TokenK::Arrow)?;

            let typath = self.require_path("return type")?;
            let tyspan = typath.span;
            let tyk = TyK::Path(Ptr::new(typath));
            let ty = Ty::new(tyk, tyspan);
            span = Span::new(span.bgn, tyspan.end);
            ret = Some(ty);
        }

        Ok(FnSig {
            params,
            ret,
            span,
        })
    }

    /// Parses a function declaration
    fn parse_function_decl(&self) -> Result<Option<Expr>, ParserError> {
        let kspan = self.token().span;

        self.eat(TokenK::KeyFn)?;
        let path = self.require_path("function name")?;

        let sig = self.parse_function_sig()?;

        let body = match self.parse_block()? {
            Some(blk) => blk,
            None => return Ok(None),
        };
        let body_expr = Expr::new(body.span, ExprK::Block(Ptr::new(body)));

        let span = Span::new(kspan.bgn, body_expr.span.end);
        let func = Fn::new(path, sig, body_expr);
        let exprk = ExprK::Fn(Ptr::new(func));
        Ok(Some(Expr::new(span, exprk)))
    }

    fn parse_function_call(&self) -> Result<Option<Expr>, ParserError> {
        let path = self.require_path("function call")?;
        let path_span = path.span;

        let args = self.parse_function_args()?;
        let args_span = Span::new(path_span.end, args.last().map(|f| f.span.end).unwrap_or(path_span.end));
        let args = args.into_iter().map(Ptr::new).collect();

        let path_exprk = ExprK::Path(Ptr::new(path));
        let call_span = Span::new(path_span.bgn, args_span.end);
        let exprk = ExprK::Call(Ptr::new(Expr::new(call_span, path_exprk)), args);


        Ok(Some(Expr::new(call_span, exprk)))
    }

    fn parse_branch(&self) -> Result<Option<Expr>, ParserError> {
        let span = self.token().span;

        self.eat(TokenK::KeyIf)?;

        let pred = self.parse_expression(0)?.ok_or_else(|| {
            ParserError::single(ParserErrorK::ExpectedExpression {
                context: "if condition",
                span: self.token().span,
            })
        })?;

        let block = match self.parse_block()? {
            Some(blk) => blk,
            None => return Ok(None),
        };

        let mut span = Span::new(span.bgn, block.span.end);

        let mut else_block = None;

        if Some(TokenK::KeyElse) == self.peek_kind(0) {
            self.eat(TokenK::KeyElse)?;
            if let Some(valid_else_block) = self.parse_block()? {
                span = Span::new(span.bgn, valid_else_block.span.end);
                else_block = Some(valid_else_block);
            }
        }

        Ok(Some(Expr::new(span,
            ExprK::If(
                Ptr::new(pred),
                Ptr::new(Expr::new(block.span, ExprK::Block(Ptr::new(block)))),
                else_block.map(|b| {
                    Ptr::new(
                        Expr::new(
                            b.span, ExprK::Block(Ptr::new(b))
                        )
                    )
                })
            )
        )))
    }

    fn parse_while_loop(&self) -> Result<Option<Expr>, ParserError> {
        let span = self.token().span;

        self.eat(TokenK::KeyWhile)?;

        let cond = self.parse_expression(0)?.ok_or_else(|| {
            ParserError::single(ParserErrorK::ExpectedExpression {
                context: "while condition",
                span: self.token().span,
            })
        })?;

        let block = match self.parse_block()? {
            Some(blk) => blk,
            None => return Ok(None),
        };

        let span = Span::new(span.bgn, block.span.end);
        let loop_node = Loop { astid: AstId::new(), kind: LoopK::While(Ptr::new(cond), Ptr::new(block)) };
        Ok(Some(Expr::new(span, ExprK::Loop(Ptr::new(loop_node)))))
    }

    fn parse_let_assignment(&self) -> Result<Option<Expr>, ParserError> {
        let let_token = self.token();
        self.eat(TokenK::KeyLet)?;

        // Parse the identifier
        let ident = self.parse_ident()?.ok_or_else(|| {
            ParserError::single(ParserErrorK::ExpectedExpression {
                context: "let binding identifier",
                span: self.token().span,
            })
        })?;

        // Optional type annotation: `: Type`
        let ty = if self.peek_kind(0) == Some(TokenK::Colon) {
            self.eat(TokenK::Colon)?;
            let typath = self.require_path("let binding type")?;
            let tyspan = typath.span;
            Ty::new(TyK::Path(Ptr::new(typath)), tyspan)
        } else {
            Ty::new(TyK::Infer, Span::none())
        };

        // Optional initializer: `= expr`
        let kind = if self.peek_kind(0) == Some(TokenK::OpEq) {
            self.advance(); // eat '='
            let init_expr = self.parse_expression(0)?.ok_or_else(|| {
                ParserError::single(ParserErrorK::ExpectedExpression {
                    context: "let binding initializer",
                    span: self.token().span,
                })
            })?;
            LocalK::Init(Ptr::new(init_expr))
        } else {
            LocalK::Decl
        };

        let span = Span::new(let_token.span.bgn, self.token().span.end);
        let local = Local {
            astid: crate::ast::AstId::new(),
            ident,
            kind,
            ty: Ptr::new(ty),
            span,
        };

        Ok(Some(Expr::new(span, ExprK::Local(Ptr::new(local)))))
    }

    fn parse_null_denotation(&self, _token: Token) -> Result<Option<Expr>, ParserError> {
        let token = self.token();
        let kind = token.kind;

        match kind {
            TokenK::LitBool |
            TokenK::LitInt |
            TokenK::LitFloat => {
                self.parse_literal()
            },
            TokenK::LitIdent => {
                if self.peek_kind(1) == Some(TokenK::LParen) {
                    self.parse_function_call()
                } else {
                    let path = self.require_path("identifier")?;
                    let span = path.span;
                    let exprk = ExprK::Path(Ptr::new(path));
                    Ok(Some(Expr::new(span, exprk)))
                }
            }
            TokenK::OpBang |
            TokenK::OpSub => {
                self.parse_prefix_op(token)
            },
            TokenK::LParen => {
                self.advance(); // eat '('
                let expr = self.parse_expression(0)?;
                self.eat(TokenK::RParen)?;
                Ok(expr)
            },
            TokenK::LBrace => {
                match self.parse_block()? {
                    Some(blk) => {
                        let span = blk.span;
                        let exprk = ExprK::Block(Ptr::new(blk));
                        Ok(Some(Expr::new(span, exprk)))
                    }
                    None => Ok(None),
                }
            },
            TokenK::RBrace => {
                Ok(None)
            },
            TokenK::Semi => {
                let span = Span::none();
                let empty = Expr::empty();
                let exprk = ExprK::Semi(Ptr::new(empty));
                Ok(Some(Expr::new(span, exprk)))
            },
            TokenK::KeyFn => {
                self.parse_function_decl()
            },
            TokenK::KeyIf => {
                self.parse_branch()
            },
            TokenK::KeyWhile => {
                self.parse_while_loop()
            },
            TokenK::KeyLet => {
                self.parse_let_assignment()
            },
            TokenK::Eof => {
                Ok(None)
            },
            _ => {
                Ok(None)
            }
        }
    }

    fn make_symbol(&self, token: &Token) -> Sym {
        let slice = &self.source[token.span.bgn..=token.span.end];
        self.symtab.make(slice)
    }
}
