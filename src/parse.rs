use std::{fmt::Display, cell::Cell};

use crate::{token::{Token, TokenK, self}, ast::{Fn, Expr, ExprK, Span, Ptr, Sym, TyK, Ty, FnSig, FnParam, Blk, Ident, Lit, LitK, BinOp, BinOpK, Path, SymTable}};



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
        let tokens: Vec<Token> = Vec::new();
        let astout: Option<Expr> = None;
        
        let mut parse = Self {
            source,
            tokens,
            astout,
        };
        
        parse.peel_tokens();
        parse.build_tree()
    }

    pub fn build_tree(&mut self) -> Result<Expr, ParserError> {
        let mut parser = Parser {
            source: self.source.clone(),
            tokens: self.tokens.clone(),
            cursor: Cell::new(0),
            symtab: SymTable::new(),
            astout: None,
            errout: None,
            change: Cell::new(true),
        };

        parser.parse()
    }

    fn peel_tokens(&mut self) {
        let source = self.source.clone();
        let chars: Vec<char> = source.chars().collect();
        let mut buff: Vec<char> = Vec::new();

        let mut iterator = chars.iter().enumerate();
        
        while let Some((curs, char)) = iterator.next() {
            if char.is_alphanumeric() | char.is_whitespace() | !char.is_multi_part() {
                self.match_key(curs, &mut buff);
            }
            
            if !char.is_numeric() & (*char != '.' ) {
                self.match_numeric(curs, &mut buff);
            }

            if !char.is_alphanumeric() & (*char != '_') {
                self.match_ident(curs, &mut buff);
            }

            if !char.is_whitespace() {
                buff.push(*char);
            }
        }

        // Try to match once more after the loop
        if !self.match_key(chars.len(), &mut buff) {
            if !self.match_ident(chars.len(), &mut buff) {
                self.match_numeric(chars.len(), &mut buff);
            }
        }
    }

    fn match_numeric(&mut self, curs: usize, buff: &mut Vec<char>) -> bool {
        if buff.is_empty() { return false; }
        let string_repr = String::from_iter(buff.iter());

        let tspan = Span {
            bgn: curs - buff.len(),
            end: curs - 1
        };
        
        let mut token = None;

        if let Ok(_) = string_repr.parse::<i64>() {
            token = Some(Token::new(tspan, TokenK::LitInt))
        };

        if token.is_none() {
            if let Ok(_) = string_repr.parse::<f64>() {
                token = Some(Token::new(tspan, TokenK::LitFloat))  
            };
        }

        if let Some(token) = token {
            self.tokens.push(token);
            buff.clear();
            return true;
        } else {
            return false;
        }
    }

    fn match_key(&mut self, curs: usize, buff: &mut Vec<char>) -> bool {
        if buff.is_empty() { return false; }
        
        let string_repr = String::from_iter(buff.iter());
        
        let tspan = Span {
            bgn: curs - buff.len(),
            end: curs - 1
        };
        
        // Handle multi-character punctuation
        let mut token = match buff.first() {
            Some('-') | Some('=') | Some(':') => {
                match buff.get(1) {
                    Some('>') => Some(Token::new(tspan, TokenK::Arrow)),
                    Some('=') => Some(Token::new(tspan, TokenK::EqEq)),
                    _ => None,
                }
            },
            Some('>') => Some(Token::new(tspan, TokenK::OpGreater)),
            Some('<') => Some(Token::new(tspan, TokenK::OpLess)),
            _ => {
                None
            }
        };

        if token.is_none() {
            token = match string_repr.as_str() {
                token::TOK_IF      => Some(Token::new(tspan, TokenK::KeyIf)),
                token::TOK_ELSE    => Some(Token::new(tspan, TokenK::KeyElse)),
                token::TOK_FN      => Some(Token::new(tspan, TokenK::KeyFn)),
                token::TOK_LET     => Some(Token::new(tspan, TokenK::KeyLet)),
                token::TOK_MUT     => Some(Token::new(tspan, TokenK::KeyMut)),
                token::TOK_REF     => Some(Token::new(tspan, TokenK::KeyRef)),
                token::TOK_WHILE   => Some(Token::new(tspan, TokenK::KeyWhile)),
                token::TOK_LOOP    => Some(Token::new(tspan, TokenK::KeyLoop)),
                token::TOK_RET     => Some(Token::new(tspan, TokenK::KeyReturn)),
                
                token::TOK_USIZE   => Some(Token::new(tspan, TokenK::Ty)),
                token::TOK_ISIZE   => Some(Token::new(tspan, TokenK::Ty)),
                token::TOK_I32     => Some(Token::new(tspan, TokenK::Ty)),
                token::TOK_I64     => Some(Token::new(tspan, TokenK::Ty)),
                token::TOK_U32     => Some(Token::new(tspan, TokenK::Ty)),
                token::TOK_U64     => Some(Token::new(tspan, TokenK::Ty)),
                token::TOK_F32     => Some(Token::new(tspan, TokenK::Ty)),
                token::TOK_F64     => Some(Token::new(tspan, TokenK::Ty)),
                
                token::TOK_COLON   => Some(Token::new(tspan, TokenK::Colon)),
                token::TOK_SEMI    => Some(Token::new(tspan, TokenK::Semi)),
                token::TOK_COMMA   => Some(Token::new(tspan, TokenK::Comma)),
                token::TOK_DOT     => Some(Token::new(tspan, TokenK::Dot)),
                token::TOK_L_BRACE => Some(Token::new(tspan, TokenK::LBrace)),
                token::TOK_R_BRACE => Some(Token::new(tspan, TokenK::RBrace)),
                token::TOK_L_PAREN => Some(Token::new(tspan, TokenK::LParen)),
                token::TOK_R_PAREN => Some(Token::new(tspan, TokenK::RParen)),
                token::TOK_L_BRACK => Some(Token::new(tspan, TokenK::LBrack)),
                token::TOK_R_BRACK => Some(Token::new(tspan, TokenK::RBrack)),
    
                token::TOK_PLUS    => Some(Token::new(tspan, TokenK::OpAdd)),
                token::TOK_MINUS   => Some(Token::new(tspan, TokenK::OpSub)),
                token::TOK_STAR    => Some(Token::new(tspan, TokenK::OpMul)),
                token::TOK_SLASH   => Some(Token::new(tspan, TokenK::OpDiv)),
                token::TOK_EQ      => Some(Token::new(tspan, TokenK::OpEq)),
    
                _ => None
            };
        }
        
        if let Some(token) = token {
            self.tokens.push(token);
            buff.clear();
            return true;
        } else {
            return false;
        }
    }
    
    fn match_ident(&mut self, curs: usize, buff: &mut Vec<char>) -> bool {
        if buff.is_empty() { return false; }
        
        let tspan = Span {
            bgn: curs - buff.len(),
            end: curs - 1
        };
        
        let matches = match buff[0] {
            c if c.is_alphabetic() | (c == '_') => {
                if buff.iter().skip(1).all(|c| {
                    c.is_alphanumeric()
                    | (*c == '_') // accept underscores in names
                }) {
                    Some(Token::new(tspan, TokenK::LitIdent))
                } else {
                    None
                }
            },
            _ => {
                None
            }
        };
        if let Some(token) = matches {
            self.tokens.push(token);
            buff.clear();
            return true;
        } else {
            return false;
        }
    }
}

trait CharImplExt {
    fn is_multi_part(&self) -> bool;
}

impl CharImplExt for char {
    fn is_multi_part(&self) -> bool {
        match self {
            '>' | '=' => true,
            _ => false
        }
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
pub enum ParserErrorK {
    Unknown,
    Unexpected,
}

#[derive(Debug, Clone)]
pub struct ParserError {
    pub errs: Vec<ParserErrorK>
}

impl ParserError {
    fn unknown() -> Self {
        Self {
            errs: vec![ParserErrorK::Unknown]
        }
    }
}

#[derive(Debug, Clone)]
pub struct Parser {
    source: String,
    tokens: Vec<Token>,
    cursor: Cell<usize>,
    symtab: SymTable,
    astout: Option<Expr>,
    errout: Option<ParserError>,
    change: Cell<bool>,
}

impl Parser {
    pub fn new() -> Self {
        Self {
            source: String::new(),
            cursor: Cell::new(0),
            tokens: Vec::new(),
            symtab: SymTable::new(),
            astout: None,
            errout: None,
            change: Cell::new(true),
        }
    }

    fn token(&self) -> Token {
        let token = self.tokens.get(self.cursor.get()).cloned();

        #[cfg(debug_assertions)]
        if self.cursor_moved() {
            if let Some(token) = &token {
                let _source = &self.source[token.span.bgn..=token.span.end];
                
                dbg_print!(blue, "CURRENT TOKEN: ");
                dbg_print!("[{}] {:?}\n", _source, token.kind);
            }
        }
        
        match token {
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

    /// Has the cursor moved since the last time this was called?
    fn cursor_moved(&self) -> bool {
        let moved = self.change.get();
        self.change.set(false);
        moved
    }

    fn advance(&self) -> Token {
        self.change.set(true);
        dbg_print!(red, "ADVANCE\n");

        self.cursor.set(self.cursor.get() + 1);
        self.token()
    }

    /// Asserts that the kind of the current token is equal to `tokenk`
    /// and advances the token pointer if it is
    fn eat(&self, tokenk: TokenK) -> Token {
        dbg_print!(red, "EAT ");
        dbg_print!("{}\n", tokenk);
        assert_eq!(self.token().kind, tokenk);
        self.advance()
    }

    /// Parse a token stream
    fn parse(&mut self) -> Result<Expr, ParserError> {
        let mut expressions = Vec::new();

        while let Some(expr) = self.parse_expression(0) {
            expressions.push(expr);
        }

        let blk = Blk::new(expressions, Span::none());
        let block = ExprK::Blk(Ptr::new(blk));
        self.astout = Some(Expr::new(Span::none(), block));

        //self.astout = self.parse_expression(0);
        
        match self.errout.take() {
            Some(err) => {
                Err(err)
            },
            None => {
                match self.astout.take() {
                    Some(ast) => {
                        Ok(ast)
                    },
                    None => {
                        Err(ParserError::unknown())
                    },
                }
            },
        }
    }
    
    /// Parses an expression using Pratt's method
    /// 
    /// Parsing begins with the token pointed to by self.token()
    fn parse_expression(&self, right_binding: usize) -> Option<Expr> {
        dbg_print!(green, "PARSE EXPRESSION "); 
        dbg_print!("{:?}\n", self.token());

        match self.token().kind {
            TokenK::Eof |
            TokenK::RBrace => {
                return None
            }
            _ => {}
        }
        
        let mut left = self.parse_null_denotation(self.token())?;
        //self.advance();
        
        // as long as the right associative binding power is less than the left associative
        // binding power of the following token, we process successive infix and suffix
        // tokens with the left denotation method. If the token is not left associative,
        // we return the null associative expression already stored in `left`
        dbg_print!(push, "");
        while right_binding < self.token().infix_binding().left() {
            dbg_print!(blue, "LEFT ASSOCIATIVE LOOP\n"); 
            left = self.parse_left_denotation(left.clone(), self.token()).unwrap_or(left);
        }

        if self.peek_kind(0) == Some(TokenK::Semi) {
            self.eat(TokenK::Semi);
        }

        dbg_print!(pop, "");
        dbg_print!(red, "RETURN EXPR\n");
        Some(left)
    }

    /// Parses a qualified or unqualified path starting with the current token
    fn parse_path(&self) -> Option<Path> {
        dbg_print!(green, "PARSE PATH\n");

        // todo: may need special handling for primitive types
        // type parsing is sort of handled here but might be
        // better in its own separate solution

        let mut list = Vec::new();
        while (self.token().kind == TokenK::LitIdent)
            | (self.token().kind == TokenK::Ty)
        {
            let ident = self.parse_ident()?;
            list.push(ident);
            
            if let Some(TokenK::ColCol) = self.peek_kind(1) {
                self.eat(TokenK::ColCol);
                self.advance();
            } else {
                break;
            }
        }
        
        let span = Span::new(
            list.first().unwrap().span.bgn,
            list.last().unwrap().span.end,
        );

        Some(Path::new(list, span))
    }
    
    fn parse_binop(&self, left: Expr, token: Token) -> Option<Expr> {
        dbg_print!(green, "PARSE BINOP\n");
        
        let kind = match token.kind {
            TokenK::OpAdd => BinOpK::Add,
            TokenK::OpSub => BinOpK::Sub,
            TokenK::OpMul => BinOpK::Mul,
            TokenK::OpDiv => BinOpK::Div,
            TokenK::OpLess => BinOpK::CmpLess,
            TokenK::OpGreater => BinOpK::CmpGreater,
            _ => {
                panic!("binop") // not a valid binop
            }
        };

        let binding = self.token().infix_binding().left();

        let lhs = left;
        self.advance();
        let rhs = self.parse_expression(binding)?;

        let opspan = token.span;
        let exspan = Span::new(lhs.span.bgn, rhs.span.end);
        let binop = BinOp { span: opspan, kind };
        let exprk = ExprK::BinOp(binop, Ptr::new(lhs), Ptr::new(rhs));
        Some(Expr::new(exspan, exprk))
    }
    
    fn parse_infix_op(&self, left: Expr, token: Token) -> Option<Expr> {
        dbg_print!(green, "PARSE INFIX OP\n");
        
        match token.kind {
            TokenK::OpGreater |
            TokenK::OpLess    |
            TokenK::OpAdd     |
            TokenK::OpSub     |
            TokenK::OpMul     |
            TokenK::OpDiv => {
                self.parse_binop(left, token)
            },
            _ => {
                panic!("infix") // not a valid infix op
            }
        }
    }

    fn parse_assignment(&self, lhs: Expr, rhs: Expr) -> Option<Expr> {
        dbg_print!(green, "PARSE ASSIGNMENT\n");
        
        match lhs.kind {
            ExprK::Path(path) => {
                let exspan = Span::new(lhs.span.bgn, rhs.span.end);
                let path = Expr::new(lhs.span, ExprK::Path(path));
                let exprk = ExprK::Assign(Ptr::new(path), Ptr::new(rhs));
                Some(Expr::new(exspan, exprk))
            },
            _ => {
                panic!("invalid lvalue") // assignment lvalue
            }
        }
    }
    
    fn parse_reverse_infix_op(&self, left: Expr, token: Token) -> Option<Expr> {
        dbg_print!(green, "PARSE REVERSE INFIX OP\n");

        self.advance();
        let lhs = left;
        let binding = self.token().infix_binding().right().saturating_sub(11);
        let rhs = self.parse_expression(binding)?;

        match token.kind {
            TokenK::OpEq => {
                self.parse_assignment(lhs, rhs)
            },
            _ => {
                panic!("reverse infix")
            }
        }
    }

    fn parse_left_denotation(&self, left: Expr, token: Token) -> Option<Expr> {
        dbg_print!(green, "PARSE LEFT DENOTATION\n");
        
        //let token = self.token();
        let kind = token.kind;

        match kind {
            TokenK::ColCol    |
            TokenK::OpGreater |
            TokenK::OpLess    |
            TokenK::OpAdd     |
            TokenK::OpSub     |
            TokenK::OpMul     |
            TokenK::OpDiv => {
                self.parse_infix_op(left, token)
            },
            TokenK::OpEq => {
                self.parse_reverse_infix_op(left, token)
            },
            TokenK::Semi => {
                let span = left.span;
                let exprk = ExprK::Semi(Ptr::new(left));
                Some(Expr::new(span, exprk))
            },
            _ => {
                panic!("left den")
            }
        }
    }


    /// Parses the current token as a literal, returning it
    /// 
    /// Leaves the cursor directly after the literal
    fn parse_literal(&self) -> Option<Expr> {
        dbg_print!(green, "PARSE LITERAL ");
        let token = self.token();

        let kind = token.kind;
        let span = token.span;
        
        let kind = match kind {
            TokenK::LitInt => LitK::Int,
            TokenK::LitFloat => LitK::Float,
            _ => {
                panic!("literal") // early return - not a literal
            }
        };

        let symbol = self.make_symbol(&token);
        let literal = Lit { symbol, kind };
        
        dbg_print!("{:?}\n", &literal);
        
        let exprk = ExprK::Lit(Ptr::new(literal));
        
        self.advance();
        Some(Expr::new(span, exprk))
    }
    
    /// Parses the current token as an ident, returning it
    /// 
    /// Leaves the cursor directly after the ident
    fn parse_ident(&self) -> Option<Ident> {
        dbg_print!(green, "PARSE IDENT\n");

        let token = self.token();
        let span = token.span;
        let sym = self.make_symbol(&token);
        let ident = Ident { name: sym, span };

        self.advance();
        Some(ident)
    }

    fn parse_prefix_op(&self, _token: Token) -> Option<Expr> {
        dbg_print!(green, "PARSE PREFIX OP\n");
        todo!();
    }

    fn parse_block(&self) -> Option<Blk> {
        dbg_print!(green, "PARSE BLOCK\n");
        dbg_print!(push, "");
        self.eat(TokenK::LBrace);

        let mut expressions = Vec::new();

        while let Some(expr) = self.parse_expression(0) {
            expressions.push(expr);
        }
        
        let span = Span::new(
            expressions.first().unwrap().span.bgn,
            expressions.last().unwrap().span.end
        );

        self.eat(TokenK::RBrace);
        
        dbg_print!(red, "RETURN BLOCK\n");
        dbg_print!(pop, "");
        Some(Blk::new(expressions, span))
    }
    
    /// Parses a functions parameters enclosed in parens.
    /// 
    /// After parsing the token cursor will be pointing to just after the right paren
    fn parse_function_params(&self) -> Option<Vec<FnParam>> {
        dbg_print!(green, "PARSE FUNCTION PARAMS\n");

        self.eat(TokenK::LParen);
        let mut params = Vec::new();
        
        while self.token().kind != TokenK::RParen {
            dbg_print!(blue, "FUNCTION PARAM LOOP\n");
            match self.token().kind {
                TokenK::Comma => {
                    self.advance();
                    continue;
                }
                TokenK::LitIdent => {
                    // the name of the parameter
                    let ident = self.parse_ident()?;
                    self.eat(TokenK::Colon);

                    // the type of the parameter
                    let typath = self.parse_path()?;
                    let tyspan = typath.span;
                    let tyk = TyK::Path(Ptr::new(typath));
                    let ty = Ty::new(tyk, tyspan);

                    let span = Span::new(ident.span.bgn, ty.span.end);
                    let param = FnParam::new(ty, ident, span);

                    params.push(param);
                },
                _ => panic!("unexpected token in function params")
            }
        }
        self.eat(TokenK::RParen);
        Some(params)
    }

    /// Parses a function signature
    /// 
    /// 
    fn parse_function_sig(&self) -> Option<FnSig> {
        let mut span = self.token().span;
        let params = self.parse_function_params()?;

        // Has a return type?
        let mut ret = None;
        if let Some(TokenK::Arrow) = self.peek_kind(0) {
            self.eat(TokenK::Arrow);

            let typath = self.parse_path()?;
            let tyspan = typath.span;
            let tyk = TyK::Path(Ptr::new(typath));
            let ty = Ty::new(tyk, tyspan);
            span = Span::new(span.bgn, tyspan.end);
            ret = Some(ty);
        }

        Some(FnSig {
            params,
            ret,
            span,
        })
    }

    /// Parses a function declaration
    /// 
    /// 
    fn parse_function_decl(&self) -> Option<Expr> {
        dbg_print!(green, "PARSE FUNCTION DECL\n");
        let kspan = self.token().span;

        self.eat(TokenK::KeyFn);
        let path = self.parse_path().expect("expected function identifier");

        let sig = self.parse_function_sig()?;

        let body = self.parse_block()?;
        let span = Span::new(kspan.bgn, body.span.end);
        
        let func = Fn::new(path, sig, body);
        let exprk = ExprK::Fn(Ptr::new(func));
        Some(Expr::new(span, exprk))
    }
    
    fn parse_branch(&self) -> Option<Expr> {
        dbg_print!(green, "PARSE BRANCH\n");
        let span = self.token().span;

        self.eat(TokenK::KeyIf);

        let pred = self.parse_expression(0).expect("expected an expression to predicate If block");
        let block = self.parse_block()?;

        let mut span = Span::new(span.bgn, block.span.end);

        let mut else_block = None;

        if Some(TokenK::KeyElse) == self.peek_kind(0) {
            self.eat(TokenK::KeyElse);
            let valid_else_block = self.parse_block()?;
            span = Span::new(span.bgn, valid_else_block.span.end);
            else_block = Some(valid_else_block);
        }

        Some(Expr::new(span,
            ExprK::If(
                Ptr::new(pred),
                Ptr::new(Expr::new(block.span, ExprK::Blk(Ptr::new(block)))),
                else_block.map(|b| { 
                    Ptr::new(
                        Expr::new(
                            b.span, ExprK::Blk(Ptr::new(b))
                        )
                    )
                })
            )
        ))
    }

    fn parse_let_assignment(&self) -> Option<Expr> {
        dbg_print!(green, "PARSE LET ASSIGNMENT\n");
        
        self.eat(TokenK::KeyLet);
        self.parse_null_denotation(self.token())
    }

    fn parse_null_denotation(&self, _token: Token) -> Option<Expr> {
        dbg_print!(green, "PARSE NULL DENOTATION ");
        dbg_print!("{:?}\n", self.token());

        let token = self.token();
        let kind = token.kind;

        match kind {
            TokenK::LitInt |
            TokenK::LitFloat => {
                self.parse_literal()
            },
            TokenK::LitIdent => {
                let path = self.parse_path().expect("expected valid path in ident");
                let span = path.span;
                let exprk = ExprK::Path(Ptr::new(path));
                Some(Expr::new(span, exprk))
            }
            TokenK::OpBang |
            TokenK::OpSub => {
                self.parse_prefix_op(token)
            },
            TokenK::LBrace => {
                let blk = self.parse_block()?;
                let span = blk.span;
                let exprk = ExprK::Blk(Ptr::new(blk));
                Some(Expr::new(span, exprk))
            },
            TokenK::RBrace => {
                None
            },
            TokenK::Semi => {
                let span = Span::none();
                let empty = Expr::empty();
                let exprk = ExprK::Semi(Ptr::new(empty));
                Some(Expr::new(span, exprk))
            },
            TokenK::KeyFn => {
                self.parse_function_decl()
            },
            TokenK::KeyIf => {
                self.parse_branch()
            },
            TokenK::KeyLet => {
                self.parse_let_assignment()
            },
            TokenK::Eof => {
                None
            },
            _ => {
                None
            }
        }
    }

    fn make_symbol(&self, token: &Token) -> Sym {
        let slice = &self.source[token.span.bgn..=token.span.end];
        self.symtab.make(slice)
    }

    fn source_fragment(&self, span: Span) -> String {
        String::from(&self.source[span.bgn..=span.end])
    }

    #[allow(dead_code)]
    #[cfg(not(debug_assertions))]
    fn dbg_state(&self) {}

    #[allow(dead_code)]
    #[cfg(debug_assertions)]
    fn dbg_state(&self) {
        println!("STATE");
        println!("  {:?}", self.token());
        println!("  '{}'", self.source_fragment(self.token().span));
    }
}
