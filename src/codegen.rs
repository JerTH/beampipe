use std::{collections::HashMap, str::FromStr, fmt::{Display, write}, cell::{RefCell, Cell}};

use crate::{ast::{*, Ident}};

#[derive(Debug, Default, Clone)]
pub enum Value {
    #[default]
    None,

    Int(i64),
    Float(f64),
    Ident(Ident),
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::None => write!(f, "()"),
            Value::Int(value) => write!(f, "{value}"),
            Value::Float(value) => write!(f, "{value}"),
            Value::Ident(value) => write!(f, "{value:?}"),
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
    vars: HashMap<Ident, Value>,
    symt: SymTable,
}

impl Eval {
    pub fn eval(expr: &Expr) -> Value {
        let mut state = Self {
            vars: Default::default(),
            symt: Default::default(),
        };
        state.eval_r(expr)
    }

    fn eval_r(&mut self, expr: &Expr) -> Value {
        let (_, _) = (expr.astid, expr.span);
        
        match &expr.kind {
            ExprK::Empty => {
                return Value::None
            },

            ExprK::Local(_) => todo!(),
            ExprK::Item(_) => todo!(),

            ExprK::Lit(lit) => {
                let value = &lit.symbol.parse::<Value>().unwrap_or_else(|_| err_sym_is_not(&lit.symbol, "integer"));
                match (&lit.kind, value) {
                    (LitK::Int, Value::Int(value)) => return Value::Int(*value),
                    (LitK::Float, Value::Float(value)) => return Value::Float(*value),
                    (lit, value) => {
                        err_is_not(value, format!("{:?}", &lit));
                    }
                }
            },

            ExprK::Blk(block) => {
                let mut returns = Value::None;
                for expr in &block.list {
                    returns = self.eval_r(expr);
                }
                if let Value::Ident(ident) = &returns {
                    if let Some(value) = self.vars.get(&ident) {
                        returns = value.clone()
                    }
                }
                returns
            },

            ExprK::Assign(lhs, rhs) => {
                match (self.eval_r(lhs), self.eval_r(rhs)) {
                    (Value::Ident(ident), value) => {
                        self.vars
                            .entry(ident)
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
                        match (self.eval_r(lhs), self.eval_r(rhs)) {
                            (Value::Int(lhs), Value::Int(rhs)) => {
                                return Value::Int(lhs + rhs);
                            },
                            (Value::Float(lhs), Value::Float(rhs)) => {
                                return Value::Float(lhs + rhs);
                            },
                            (lhs, rhs) => {
                                err_op_mismatch("add", lhs, rhs);
                            }
                        }
                    },
                    BinOpK::Sub => {
                        match (self.eval_r(lhs), self.eval_r(rhs)) {
                            (Value::Int(lhs), Value::Int(rhs)) => {
                                return Value::Int(lhs - rhs);
                            },
                            (Value::Float(lhs), Value::Float(rhs)) => {
                                return Value::Float(lhs - rhs);
                            },
                            (lhs, rhs) => {
                                err_op_mismatch("subtract", lhs, rhs);
                            }
                        }
                    },
                    BinOpK::Div => {
                        match (self.eval_r(lhs), self.eval_r(rhs)) {
                            (Value::Int(lhs), Value::Int(rhs)) => {
                                return Value::Int(lhs / rhs);
                            },
                            (Value::Float(lhs), Value::Float(rhs)) => {
                                return Value::Float(lhs / rhs);
                            },
                            (lhs, rhs) => {
                                err_op_mismatch("divide", lhs, rhs);
                            }
                        }
                    },
                    BinOpK::Mul => {
                        match (self.eval_r(lhs), self.eval_r(rhs)) {
                            (Value::Int(lhs), Value::Int(rhs)) => {
                                return Value::Int(lhs * rhs);
                            },
                            (Value::Float(lhs), Value::Float(rhs)) => {
                                return Value::Float(lhs * rhs);
                            },
                            (lhs, rhs) => {
                                err_op_mismatch("multiply", lhs, rhs);
                            }
                        }
                    },
                }
            },

            ExprK::AssignOp(_, _, _) => { todo!() },

            ExprK::Path(path) => {
                let full_name = path.list.iter().fold(String::new(), |acc, x| {
                    format!("{}::{}", acc, x)
                });
                let ident = Ident {
                    name: self.symt.make(full_name),
                    span: Span::none(),
                };
                return Value::Ident(ident)
            },
        }
    }
}

#[derive(Debug)]
pub enum Marker {
    Symbol(Sym),
    Ident(Ident),
    Offset(usize),
}

impl Display for Marker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Marker::Ident(ident) => write!(f, "{}", ident),
            Marker::Offset(offset) => write!(f, "${}", offset),
            Marker::Symbol(sym) => write!(f, "&{}", sym.as_string()),
        }
    }
}

#[derive(Debug)]
pub enum Ir {
    Nop,
    Add,
    Sub,
    Div,
    Mul,
    Store(Marker),
    Bool(bool),
    Integer(i64),
    Float(f64),
    Jump(Marker),
    Load(Marker),
}

impl Display for Ir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
    fn emit(&self, ir: Ir) {
        println!("ircode::emit: {}", ir);
        self.code.borrow_mut().push(ir);
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
        println!("emit_r: {:?}", self.code);

        let (_, _) = (expr.astid, expr.span);
        let code = &self.code;
        let syms = &self.syms;

        match &expr.kind {
            ExprK::Empty => {
                code.emit(Ir::Nop)
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
            ExprK::Blk(block) => {
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
                }
            },
            ExprK::AssignOp(_, _, _) => todo!(),
            ExprK::Path(path) => {
                match Eval::eval(&expr) {
                    Value::None => {},
                    Value::Int(value) => todo!(),
                    Value::Float(value) => todo!(),
                    Value::Ident(ident) => {
                        code.emit(Ir::Load(Marker::Ident(ident)))
                    },
                }
            },
        }
    }
}

const TOK_IF: &'static str = "if";
const TOK_FN: &'static str = "fn";
const TOK_LET: &'static str = "let";
const TOK_MUT: &'static str = "mut";
const TOK_REF: &'static str = "ref";
const TOK_WHILE: &'static str = "while";
const TOK_LOOP: &'static str = "loop";
const TOK_RET: &'static str = "return";

const TOK_L_PAREN: &'static str = "(";
const TOK_R_PAREN: &'static str = ")";
const TOK_L_BRACE: &'static str = "{";
const TOK_R_BRACE: &'static str = "}";
const TOK_L_BRACK: &'static str = "[";
const TOK_R_BRACK: &'static str = "]";

const TOK_ARROW: &'static str = "->";
const TOK_PLUS: &'static str = "+";
const TOK_MINUS: &'static str = "-";
const TOK_STAR: &'static str = "*";
const TOK_SLASH: &'static str = "/";
const TOK_COLON: &'static str = ":";
const TOK_SEMI: &'static str = ";";
const TOK_COMMA: &'static str = ",";
const TOK_DOT: &'static str = ".";

const TOK_USIZE: &'static str = "usize";
const TOK_U32: &'static str = "u32";
const TOK_U64: &'static str = "u64";
const TOK_F32: &'static str = "f32";
const TOK_F64: &'static str = "f64";

#[derive(Debug, Clone)]
pub enum TokenK {
    KeyFn,
    KeyReturn,
    KeyLet,

    OpAdd,
    OpSub,
    OpMul,
    OpDiv,

    LitIdent,
    LitInt,
    LitFloat,

    Ty,
    
    LParen,
    RParen,
    LBrack,
    RBrack,
    LBrace,
    RBrace,
    Semi,
    Dot,
    Arrow,
    Colon,
    Comma,

    // .
    EqEq,
    KeyIf,
    KeyMut,
    KeyRef,
    KeyWhile,
    KeyLoop,
}

impl Display for TokenK {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(&format!("{:?}", self))
    }
}

enum SemanticK {
    None,
    Lit,
    BinOp,
}

#[derive(Debug, Clone)]
pub struct Token {
    kind: TokenK,
    span: Span,
}

impl Token {
    fn new(span: Span, kind: TokenK) -> Self {
        Self {
            kind,
            span,
        }
    }

    fn semantic_kind(&self) -> SemanticK {
        match self.kind {
            TokenK::OpAdd |
            TokenK::OpSub |
            TokenK::OpDiv |
            TokenK::OpMul => {
                SemanticK::BinOp
            },
            _ => {
                SemanticK::None
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct Parse {
    source: String, // store a copy for debug purposes
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
    pub fn parse<S: Into<String>>(source: S) -> Expr {
        let source: String = source.into();
        let tokens: Vec<Token> = Vec::new();
        let astout: Option<Expr> = None;
        
        let mut parse = Self {
            source,
            tokens,
            astout,
        };
        
        parse.peel_tokens();

        parse.build_tree();
        
        dbg!(&parse.astout);

        match parse.astout {
            Some(ast) => {
                let out = Emit::emit(&ast);
                println!("\n\n{}\n\n", out);
            },
            None => {
                panic!("failed to build ast")
            },
        }

        todo!()
    }
    
    pub fn build_tree(&mut self) {
        let mut parser = Parser {
            source: self.source.clone(),
            tokens: self.tokens.clone(),
            cursor: Cell::new(0),
            rewind: RefCell::new(Vec::new()),
            expect: Vec::new(),
            symtab: SymTable::new(),
            astout: None,
            errout: None,
        };

        match parser.parse() {
            Ok(ast) => self.astout = Some(ast),
            Err(err) => panic!("parser err: {err:?}"),
        }
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
            
            if !char.is_numeric() {
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
            self.match_ident(chars.len(), &mut buff);
        }

        println!("{buff:?}");
        println!("{}", self);
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

        if let Ok(_) = string_repr.parse::<f64>() {
            token = Some(Token::new(tspan, TokenK::LitFloat))  
        };

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
            _ => {
                None
            }
        };

        if token.is_none() {
            token = match string_repr.as_str() {
                TOK_IF      => Some(Token::new(tspan, TokenK::KeyIf)),
                TOK_FN      => Some(Token::new(tspan, TokenK::KeyFn)),
                TOK_LET     => Some(Token::new(tspan, TokenK::KeyLet)),
                TOK_MUT     => Some(Token::new(tspan, TokenK::KeyMut)),
                TOK_REF     => Some(Token::new(tspan, TokenK::KeyRef)),
                TOK_WHILE   => Some(Token::new(tspan, TokenK::KeyWhile)),
                TOK_LOOP    => Some(Token::new(tspan, TokenK::KeyLoop)),
                TOK_RET     => Some(Token::new(tspan, TokenK::KeyReturn)),
                
                TOK_USIZE   => Some(Token::new(tspan, TokenK::Ty)),
                TOK_U32     => Some(Token::new(tspan, TokenK::Ty)),
                TOK_U64     => Some(Token::new(tspan, TokenK::Ty)),
                TOK_F32     => Some(Token::new(tspan, TokenK::Ty)),
                TOK_F64     => Some(Token::new(tspan, TokenK::Ty)),
    
                TOK_COLON   => Some(Token::new(tspan, TokenK::Colon)),
                TOK_SEMI    => Some(Token::new(tspan, TokenK::Semi)),
                TOK_COMMA   => Some(Token::new(tspan, TokenK::Comma)),
                TOK_DOT     => Some(Token::new(tspan, TokenK::Dot)),
                TOK_L_BRACE => Some(Token::new(tspan, TokenK::LBrace)),
                TOK_R_BRACE => Some(Token::new(tspan, TokenK::RBrace)),
                TOK_L_PAREN => Some(Token::new(tspan, TokenK::LParen)),
                TOK_R_PAREN => Some(Token::new(tspan, TokenK::RParen)),
                TOK_L_BRACK => Some(Token::new(tspan, TokenK::LBrack)),
                TOK_R_BRACK => Some(Token::new(tspan, TokenK::RBrack)),
    
                TOK_PLUS    => Some(Token::new(tspan, TokenK::OpAdd)),
                TOK_MINUS   => Some(Token::new(tspan, TokenK::OpSub)),
                TOK_STAR    => Some(Token::new(tspan, TokenK::OpMul)),
                TOK_SLASH   => Some(Token::new(tspan, TokenK::OpDiv)),
    
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

#[derive(Debug, Clone)]
enum ParserErrorK {
    Unknown,
    Unexpected,
}

#[derive(Debug, Clone)]
struct ParserError {
    errs: Vec<ParserErrorK>
}

impl ParserError {
    fn unknown() -> Self {
        Self {
            errs: vec![ParserErrorK::Unknown]
        }
    }

    fn unexpected() -> ParserError {
        Self {
            errs: vec![ParserErrorK::Unexpected]
        }
    }
}

#[derive(Debug, Clone)]
struct Parser {
    source: String,
    tokens: Vec<Token>,
    cursor: Cell<usize>,
    rewind: RefCell<Vec<usize>>,
    expect: Vec<Token>,
    symtab: SymTable,
    astout: Option<Expr>,
    errout: Option<ParserError>,
}

impl Parser {
    fn new() -> Self {
        Self {
            source: String::new(),
            cursor: Cell::new(0),
            rewind: RefCell::new(Vec::new()),
            tokens: Vec::new(),
            expect: Vec::new(),
            symtab: SymTable::new(),
            astout: None,
            errout: None,
        }
    }

    fn token(&self) -> Option<Token> {
        self.tokens.get(self.cursor.get()).cloned()
    }

    fn advance(&self) -> Option<Token> {
        self.cursor.set(self.cursor.get() + 1);
        self.token()
    }

    fn save_cursor(&self) {
        self.rewind.borrow_mut().push(self.cursor.get());
    }

    fn rewind_cursor(&self) {
        if let Some(rewind) = self.rewind.borrow_mut().pop() {
            self.cursor.set(rewind)
        }
    }

    fn parse(&mut self) -> Result<Expr, ParserError> {
        println!("parse");

        self.astout = self.parse_expr(0);
        
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

    fn parse_expr(&mut self, precedence: usize) -> Option<Expr> {
        println!("parse_expr");

        self.parse_binop(precedence)
    }

    fn parse_binop(&mut self, precedence: usize) -> Option<Expr> {
        println!("parse_binop");

        self.save_cursor();

        let mut span: Span = Span::none();
        let mut lhso: Option<Expr> = None;

        if let Some(lhst) = self.token() {
            println!("\ttok{:?}", &lhst);

            let lhs = match lhst.kind {
                TokenK::LitInt => Some(Expr::new(lhst.span, self.make_exprk(&lhst))),
                TokenK::LitFloat => Some(Expr::new(lhst.span, self.make_exprk(&lhst))),
                TokenK::LitIdent => Some(Expr::new(lhst.span, self.make_exprk(&lhst))),
                _ => None,
            };
            span = lhst.span;
            lhso = lhs;
        }
        
        if let Some(lhs) = lhso.clone() {
            println!("\texpr{:?}", &lhs);

            loop {
                if let Some((op, binding)) = self.parse_infix_op() {

                    if binding.lhs() < precedence {
                        break;
                    }

                    if let Some(rhst) = self.advance() {
                        if let Some(rhs) = self.parse_expr(binding.rhs()) {
                            if let ExprK::BinOp(op, _, _) = op.kind {
                                span.end = rhst.span.end;
                                lhso = Some(Expr::new(span, ExprK::BinOp(op.clone(), Ptr::new(lhs.clone()), Ptr::new(rhs))));
                            };
                        }
                    }
                } else {
                    break;
                }
            }
        }

        if lhso.is_none() {
            self.rewind_cursor()
        }

        lhso
    }

    fn parse_infix_op(&mut self) -> Option<(Expr, (usize, usize))> {
        println!("parse_infix_op");

        self.save_cursor();

        let mut op = None;
        if let Some(opt) = self.advance() {
            op = match opt.kind {
                TokenK::OpAdd => {
                    Some(Expr::new(opt.span, ExprK::binop(opt.span, BinOpK::Add)))
                },
                TokenK::OpSub => {
                    Some(Expr::new(opt.span, ExprK::binop(opt.span, BinOpK::Sub)))
                },
                TokenK::OpMul => {
                    Some(Expr::new(opt.span, ExprK::binop(opt.span, BinOpK::Mul)))
                },
                TokenK::OpDiv => {
                    Some(Expr::new(opt.span, ExprK::binop(opt.span, BinOpK::Div)))
                },
                _ => None
            };
        }

        match op {
            Some(op) => {
                Some((op, self.infix_binding()))
            },
            None => {
                self.rewind_cursor();
                return None
            },
        }
    }

    fn make_exprk(&self, token: &Token) -> ExprK {
        match token.kind {
            TokenK::LitInt => ExprK::Lit(Ptr::new(Lit { symbol: self.make_symbol(&token), kind: LitK::Int })),
            TokenK::LitFloat => ExprK::Lit(Ptr::new(Lit { symbol: self.make_symbol(&token), kind: LitK::Float })),
            TokenK::LitIdent => {
                ExprK::Path(Ptr::new(Path {
                    astid: AstId::new(),
                    list: vec![Ident {
                        name: self.symtab.make(self.source_fragment(token.span)),
                        span: token.span,
                    }],
                    span: token.span,
                }))
            },
            TokenK::OpMul => {
                ExprK::BinOp(
                    BinOp { span: token.span, kind: BinOpK::Mul },
                    Ptr::new(Expr::new(token.span, ExprK::Empty)),
                    Ptr::new(Expr::new(token.span, ExprK::Empty))
                )
            },
            TokenK::OpDiv => {
                ExprK::BinOp(
                    BinOp { span: token.span, kind: BinOpK::Mul },
                    Ptr::new(Expr::new(token.span, ExprK::Empty)),
                    Ptr::new(Expr::new(token.span, ExprK::Empty))
                )
            },
            _ => ExprK::Empty
        }
    }

    fn make_symbol(&self, token: &Token) -> Sym {
        let slice = &self.source[token.span.bgn..=token.span.end];
        self.symtab.make(slice)
    }
    
    fn prefix_binding(&self) -> ((), usize) {
        match self.token() {
            Some(token) => {
                match token.kind {
                    TokenK::OpAdd =>    ((), 5),
                    TokenK::OpSub =>    ((), 5),
                    _ =>                ((), 1)
                }
            },
            None => {
                ((), 1)
            }
        }
    }

    fn infix_binding(&self) -> (usize, usize) {
        match self.token() {
            Some(token) => {
                match token.kind {
                    TokenK::OpAdd =>    (1, 2),
                    TokenK::OpSub =>    (1, 2),
                    TokenK::OpMul =>    (3, 4),
                    TokenK::OpDiv =>    (3, 4),
                    TokenK::Dot =>      (6, 5),
                    _ =>                (0, 0),
                }
            },
            None => {
                (0, 0)
            },
        }
    }
    
    fn postfix_binding(&self) -> (usize, ()) {
        match self.token() {
            _ =>                (1, ())
        }
    }

    fn source_fragment(&self, span: Span) -> String {
        String::from(&self.source[span.bgn..=span.end])
    }
}

trait BindingExt<L, R> {
    fn lhs(&self) -> L;
    fn rhs(&self) -> R;
}

impl<L, R> BindingExt<L, R> for (L, R)
where
    L: Clone,
    R: Clone
{
    fn lhs(&self) -> L {
        self.0.clone()
    }

    fn rhs(&self) -> R {
        self.1.clone()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_parser() {
        let source = include_str!("code.rs");
        let result = Parse::parse(source);
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

fn err_unexpected_char(unexpected: char) -> ! {
    panic!("unexpected character: {unexpected}");
}

fn err_malformed_token(buffer: &Vec<char>) -> ! {
    let string = String::from_iter(buffer.into_iter());
    panic!("malformed token: {string}");
}

//fn foo(a: i32, b: f32) -> usize {
//    if a == 0 {
//        return b * 2.0 + baz(a);
//    }
//}
