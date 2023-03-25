use std::{collections::HashMap, str::FromStr, fmt::{Display, write}, cell::{RefCell, Cell}, borrow::Borrow};

use crate::{ast::{*, Ident}};

static DBG_INDENT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

macro_rules! dbg_print {
    (push, $($t:tt)*) => {
        {
            DBG_INDENT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        dbg_print_color!($($t)*)
    };
    
    (pop, $($t:tt)*) => {
        {
            DBG_INDENT.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        }
        dbg_print_color!($($t)*)
    };

    ($($t:tt)*) => {
        dbg_print_color!($($t)*)
    };
}

macro_rules! dbg_print_color {
    (red, $string:tt) => {
        #[cfg(debug_assertions)] dbg_print_indent!("\x1b[1;31m{}\x1b[0m", $string)
    };
    
    (green, $string:tt) => {
        #[cfg(debug_assertions)] dbg_print_indent!("\x1b[1;32m{}\x1b[0m", $string)
    };

    (yellow, $string:tt) => {
        #[cfg(debug_assertions)] dbg_print_indent!("\x1b[1;33m{}\x1b[0m", $string)
    };

    (blue, $string:tt) => {
        #[cfg(debug_assertions)] dbg_print_indent!("\x1b[1;34m{}\x1b[0m", $string)
    };

    (magenta, $string:tt) => {
        #[cfg(debug_assertions)] dbg_print_indent!("\x1b[1;35m{}\x1b[0m", $string)
    };

    ($($t:tt)*) => {
        #[cfg(debug_assertions)] print!($($t)*)
    };
}

macro_rules! dbg_print_indent {
    ($($t:tt)*) => {
        #[cfg(debug_assertions)] {
            print!("{}", "   ".repeat(DBG_INDENT.load(std::sync::atomic::Ordering::SeqCst)));
            print!($($t)*);
        }
    };
}

fn foo() {
    print!("{}", 0)
}
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
        #[cfg(debug_assertions)] println!("ircode::emit: {}", ir);
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

#[derive(Debug, Clone, Copy)]
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

    // generated - sort these
    EqEq,
    KeyIf,
    KeyMut,
    KeyRef,
    KeyWhile,
    KeyLoop,
    Eof,
    ColCol,
}

impl Display for TokenK {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(&format!("{:?}", self))
    }
}

#[derive(Debug, Clone)]
pub struct Token {
    kind: TokenK,
    span: Span,
}
// ==========================================================================================================
// =====================================================================================================TOKEN
// ==========================================================================================================
impl Token {
    pub fn new(span: Span, kind: TokenK) -> Self {
        Self {
            kind,
            span,
        }
    }

    pub fn prefix_binding(&self) -> ((), usize) {
        match self.kind {
            TokenK::OpAdd   => ((), 50),
            TokenK::OpSub   => ((), 50),
            _               => ((), 00)
        }
    }
    
    pub fn infix_binding(&self) -> (usize, usize) {
        match self.kind {
            TokenK::OpAdd   => (50, 51),
            TokenK::OpSub   => (50, 51),
            TokenK::OpMul   => (60, 61),
            TokenK::OpDiv   => (60, 61),
            TokenK::Dot     => (80, 81),
            TokenK::ColCol  => (90, 91),
            _               => (00, 00),
        }
    }
    
    pub fn postfix_binding(&self) -> (usize, ()) {
        match self.kind {
            _               => (0, ())
        }
    }
    
    pub fn null_denotation(&self) -> fn() -> Expr {
        match self.kind {
            _ => {
                || { Expr::empty() }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum ParserErrorK {
    Unknown,
    Unexpected,
}

#[derive(Debug, Clone)]
pub struct ParserError {
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
pub struct Parser {
    source: String,
    tokens: Vec<Token>,
    cursor: Cell<usize>,
    rewind: RefCell<Vec<usize>>,
    expect: Vec<Token>,
    symtab: SymTable,
    astout: Option<Expr>,
    errout: Option<ParserError>,
    change: Cell<bool>,
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
            change: Cell::new(true),
        }
    }

    fn token(&self) -> Token {
        let token = self.tokens.get(self.cursor.get()).cloned();

        #[cfg(debug_assertions)]
        if self.cursor_moved() {
            if let Some(token) = &token {
                let source = &self.source[token.span.bgn..=token.span.end];
                
                dbg_print!(blue, "CURRENT TOKEN: ");
                dbg_print!("[{}] {:?}\n", source, token.kind);
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

    fn peek(&self, offset: isize) -> Option<Token> {
        let offset_cursor = self.cursor.get().checked_add_signed(offset)?;
        self.tokens.get(offset_cursor).cloned()
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
    
    fn save_cursor(&self) {
        dbg_print!(green, "SAVE ");
        dbg_print!("{} / {}\n", self.cursor.get(), self.tokens.len());

        self.rewind.borrow_mut().push(self.cursor.get());
    }

    fn rewind_cursor(&self) {
        if let Some(rewind) = self.rewind.borrow_mut().pop() {
            dbg_print!(magenta, "REWIND ");
            dbg_print!("{} <- {}\n", rewind, self.cursor.get());
            self.change.set(true);
            self.cursor.set(rewind)
        } else {
            dbg_print!(magenta, "REWIND: NO REWIND HISTORY\n");
        }
    }

    fn parse(&mut self) -> Result<Expr, ParserError> {
        self.astout = self.parse_expression(0);

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
        dbg_print!(green, "PARSE EXPRESSION\n");

        let mut left = self.parse_null_denotation(self.token())
            .expect("expected null associative");
        self.advance();

        // as long as the right associative binding power is less than the left associative
        // binding power of the following token, we process successive infix and suffix
        // tokens with the left denotation method. If the token is not left associative,
        // we return the null associative expression already stored in `left`
        while right_binding < self.token().infix_binding().left() {
            dbg_print!(push, blue, "LOOP\n");

            left = self.parse_left_denotation(left.clone(), self.token()).unwrap_or(left);
        }

        dbg_print!(red, "RETURN EXPR\n");
        dbg_print!(pop, "");
        Some(left)
    }
    
    /// Combines paths
    fn parse_path(&self, lhs: Expr, rhs: Expr) -> Option<Expr> {
        dbg_print!(green, "PARSE PATH\n");

        match (lhs.kind, rhs.kind) {
            (ExprK::Path(lhs), ExprK::Path(rhs)) => {
                let list = [&lhs.list[..], &rhs.list[..]].concat();
                let span = Span::new(lhs.span.bgn, rhs.span.end);
                let path = Path::new(list, span);
                let exprk = ExprK::Path(Ptr::new(path));
                Some(Expr::new(span, exprk))
            },
            _ => {
                None
            }
        }
    }
    
    fn parse_binop(&self, left: Expr, token: Token) -> Option<Expr> {
        dbg_print!(green, "PARSE BINOP\n");
        
        let kind = match token.kind {
            TokenK::OpAdd => BinOpK::Add,
            TokenK::OpSub => BinOpK::Sub,
            TokenK::OpMul => BinOpK::Mul,
            TokenK::OpDiv => BinOpK::Div,
            _ => return panic!("binop") // early return - not a valid binop
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
            TokenK::OpAdd |
            TokenK::OpSub |
            TokenK::OpMul |
            TokenK::OpDiv => {
                self.parse_binop(left, token)
            },
            TokenK::ColCol => {
                let binding = self.token().infix_binding().right();
                let lhs = left;
                let rhs = self.parse_expression(binding)?;
                self.parse_path(lhs, rhs)
            }
            _ => panic!("infix") // early return - not a valid binop
        }
    }

    fn parse_left_denotation(&self, left: Expr, token: Token) -> Option<Expr> {
        dbg_print!(green, "PARSE LEFT DENOTATION\n");
        
        //let token = self.token();
        let kind = token.kind;

        match kind {
            TokenK::ColCol |
            TokenK::OpAdd  |
            TokenK::OpSub  |
            TokenK::OpMul  |
            TokenK::OpDiv => {
                self.parse_infix_op(left, token)
            }
            _ => {
                panic!("left den")
            }
        }
    }

    fn parse_literal(&self, token: Token) -> Option<Expr> {
        dbg_print!(green, "PARSE LITERAL ");
        
        //let token = self.token();
        let kind = token.kind;
        let span = token.span;
        
        let kind = match kind {
            TokenK::LitInt => LitK::Int,
            TokenK::LitFloat => LitK::Float,
            _ => {
                return None // early return - not a literal
            }
        };

        let symbol = self.make_symbol(&token);
        let literal = Lit { symbol, kind };
        
        dbg_print!("{:?}\n", &literal);
        
        let exprk = ExprK::Lit(Ptr::new(literal));
        Some(Expr::new(span, exprk))
    }
    
    /// Parses an ident as a Path, paths can then be combined
    fn parse_ident(&self, token: Token) -> Option<Expr> {
        dbg_print!(green, "PARSE IDENT\n");
        
        //let token = self.token();

        let span = token.span;
        let name = self.make_symbol(&token);
        let ident = Ident { name, span };
        let path = Path::new(vec![ident], span);
        let exprk = ExprK::Path(Ptr::new(path));
        Some(Expr::new(span, exprk))
    }

    fn parse_null_denotation(&self, token: Token) -> Option<Expr> {
        dbg_print!(green, "PARSE NULL DENOTATION ");
        dbg_print!("{:?}\n", token);

        //let token = self.token();        
        let kind = token.kind;

        match kind {
            TokenK::LitInt |
            TokenK::LitFloat => {
                self.parse_literal(token)
            },
            TokenK::LitIdent => {
                self.parse_ident(token)
            }
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
}


#[derive(Debug, Default)]
pub struct Parse {
    source: String,
    tokens: Vec<Token>,
    astout: Option<Expr>,
}

impl Display for Parse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return Ok(());

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
            rewind: RefCell::new(Vec::new()),
            expect: Vec::new(),
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_parser() {
        let source = include_str!("example/expr.rs");
        let expr = Parse::parse(source).expect("expected ast");
        let code = Emit::emit(&expr);
        let result = Eval::eval(&expr);

        println!("{}", source);
        println!("{}", code);
        println!("{}", result);
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
