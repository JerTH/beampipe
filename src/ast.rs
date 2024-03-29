use std::{
    collections::HashMap,
    error::Error,
    fmt::{Debug, Display},
    hash::Hash,
    ops::{Bound, Deref, DerefMut, RangeBounds},
    str::FromStr,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, RwLock,
    },
};

type PtrType<T> = Box<T>;

#[derive(Clone)]
pub struct Ptr<T: ?Sized>(PtrType<T>);

impl<T> Ptr<T> {
    pub fn new(val: T) -> Self {
        Ptr(PtrType::new(val))
    }
}

impl<T> Deref for Ptr<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> Debug for Ptr<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            f.write_fmt(format_args!("{:#?}", self.0.deref()))
        } else {
            f.write_fmt(format_args!("{:?}", self.0.deref()))
        }
    }
}

impl<T> Display for Ptr<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

macro_rules! p {
    ($e:expr) => {
        Ptr::new($e)
    };
}

static AST_ID_COUNT: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AstId(usize);

impl AstId {
    pub fn new() -> Self {
        Default::default()
    }
}

impl Default for AstId {
    fn default() -> Self {
        AstId(AST_ID_COUNT.fetch_add(1, Ordering::SeqCst))
    }
}

impl Deref for AstId {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for AstId {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

macro_rules! astid {
    () => {
        AstId::new()
    };
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Span {
    pub bgn: usize,
    pub end: usize,
}

impl Span {
    pub fn none() -> Self {
        Span { bgn: 0, end: 0 }
    }

    pub fn new(bgn: usize, end: usize) -> Self {
        Span { bgn, end }
    }
}

impl RangeBounds<usize> for Span {
    fn start_bound(&self) -> Bound<&usize> {
        Bound::Included(&self.bgn)
    }

    fn end_bound(&self) -> Bound<&usize> {
        Bound::Included(&self.end)
    }
}

impl Debug for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} -> {}", self.bgn, self.end)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Path {
    pub astid: AstId,
    pub list: Vec<Ident>,
    pub span: Span,
}

impl Path {
    pub fn new(list: Vec<Ident>, span: Span) -> Self {
        Self {
            astid: astid!(),
            list,
            span,
        }
    }
}

impl From<&Ptr<Path>> for String {
    fn from(value: &Ptr<Path>) -> Self {
        String::from(&*value.0)
    }
}

impl From<Ptr<Path>> for String {
    fn from(value: Ptr<Path>) -> Self {
        String::from(&*value.0)
    }
}

impl From<&Path> for String {
    fn from(value: &Path) -> Self {
        value.list
            .iter()
            .fold(String::new(), |acc, x| format!("{}::{}", acc, x))
            .trim_start_matches("::")
            .into()
    }
}

#[derive(Debug, Clone)]
pub struct Ty {
    pub astid: AstId,
    pub kind: TyK,
    pub span: Span,
}

impl Ty {
    pub fn new(kind: TyK, span: Span) -> Self {
        Self {
            astid: astid!(),
            kind,
            span,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Local {
    pub astid: AstId,
    pub ident: Ident,
    pub kind: LocalK,
    pub ty: Ptr<Ty>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Blk {
    pub astid: AstId,
    pub list: Vec<Ptr<Expr>>,
    pub span: Span,
}

impl Blk {
    pub fn new(list: Vec<Ptr<Expr>>, span: Span) -> Self {
        Blk {
            astid: astid!(),
            list,
            span,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Loop {
    pub astid: AstId,
    pub kind: LoopK,
}

#[derive(Debug, Clone)]
pub enum LoopK {
    For(Ptr<Expr>, Ptr<Blk>),
    While(Ptr<Expr>, Ptr<Blk>),
    Loop(Option<Ident>, Ptr<Blk>),
}

pub struct Call {
    pub astid: AstId,
    pub path: Path,
    pub args: Vec<Expr>,
}

impl Call {
    pub fn new(path: Path, args: Vec<Expr>, span: Span) -> Self {
        Call {
            astid: astid!(),
            path,
            args
        }
    }
}

#[derive(Debug, Clone)]
pub struct Item {
    pub astid: AstId,
    pub ident: Ident,
}

#[derive(Debug, Clone)]
pub struct Lit {
    pub symbol: Sym,
    pub kind: LitK,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ident {
    pub name: Sym,
    pub span: Span,
}

impl Ident {
    pub fn as_string(&self) -> String {
        self.name.as_string()
    }
}

impl Display for Ident {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name.as_string())
    }
}

#[derive(Debug, Clone)]
pub struct BinOp {
    pub span: Span,
    pub kind: BinOpK,
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub astid: AstId,
    pub kind: ExprK,
    pub span: Span,
}

impl Expr {
    pub fn new(span: Span, exprk: ExprK) -> Self {
        Expr {
            astid: astid!(),
            kind: exprk,
            span,
        }
    }

    pub fn empty() -> Expr {
        Expr {
            astid: astid!(),
            kind: ExprK::Empty,
            span: Span::none(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FnArg {
    pub id: AstId,
    pub expr: Expr,
    pub span: Span,
}

impl FnArg {
    pub fn new(expr: Expr, span: Span) -> Self {
        FnArg {
            id: astid!(),
            expr,
            span
        }
    }
}

#[derive(Debug, Clone)]
pub struct FnParam {
    pub id: AstId,
    pub ty: Ty,
    pub ident: Ident,
    pub span: Span,
}

impl FnParam {
    pub fn new(ty: Ty, ident: Ident, span: Span) -> Self {
        FnParam {
            id: astid!(),
            ty,
            ident,
            span,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FnSig {
    pub params: Vec<FnParam>,
    pub ret: Option<Ty>,
    pub span: Span,
}

impl FnSig {
    pub fn new(params: Vec<FnParam>, ret: Option<Ty>, span: Span) -> Self {
        Self { params, span, ret }
    }
}

#[derive(Debug, Clone)]
pub struct Fn {
    pub id: AstId,
    pub path: Path,
    pub sig: FnSig,
    pub body: Ptr<Expr>,
}

impl Fn {
    pub fn new(path: Path, sig: FnSig, body: Expr) -> Self {
        Self {
            id: astid!(),
            path,
            sig,
            body: Ptr::new(body),
        }
    }
}

// Kinds
#[derive(Debug, Clone)]
pub enum TyK {
    Array(Ptr<Ty>),
    Path(Ptr<Path>),
    Infer,
}

#[derive(Debug, Clone)]
pub enum LocalK {
    Decl,
    Init(Ptr<Expr>),
}

#[derive(Debug, Clone)]
pub enum ItemK {}

#[derive(Debug, Clone)]
pub enum LitK {
    Bool,
    Int,
    Float,
}

#[derive(Debug, Clone)]
pub enum BinOpK {
    Add,
    Sub,
    Div,
    Mul,
    CmpLess,
    CmpGreater,
}

#[derive(Debug, Clone)]
pub enum ExprK {
    Empty,
    Item(Ptr<Item>),
    Semi(Ptr<Expr>), // semicolon terminated expression
    Local(Ptr<Local>),
    Lit(Ptr<Lit>),
    Block(Ptr<Blk>),
    Loop(Ptr<Loop>),
    Assign(Ptr<Expr>, Ptr<Expr>),
    BinOp(BinOp, Ptr<Expr>, Ptr<Expr>),
    AssignOp(Ptr<BinOp>, Ptr<Expr>, Ptr<Expr>),
    Path(Ptr<Path>),
    Fn(Ptr<Fn>),
    If(Ptr<Expr>, Ptr<Expr>, Option<Ptr<Expr>>),
    Call(Ptr<Expr>, Vec<Ptr<FnArg>>),
}

impl ExprK {
    pub fn binop(span: Span, kind: BinOpK) -> Self {
        ExprK::BinOp(
            BinOp { span, kind },
            p!(Expr::new(span, ExprK::Empty)),
            p!(Expr::new(span, ExprK::Empty)),
        )
    }
}

#[derive(Default, Debug, Clone)]
pub struct SymTable {
    astid: AstId,
    table: Arc<RwLock<HashMap<u64, String>>>,
}

impl SymTable {
    pub fn new() -> Self {
        Self {
            table: Arc::new(RwLock::new(HashMap::new())),
            astid: astid!(),
        }
    }

    pub fn make<'a, S: Into<String>>(&self, string: S) -> Sym {
        let string: String = string.into();
        
        match self.table.write() {
            Ok(mut guard) => {
                let key = SymTable::hash_str(string.as_str());
                if let Some(item) = guard.get(&key) {
                    
                } else {
                    guard.insert(key, string.into());
                }

                Sym {
                    index: key,
                    table: self.clone(),
                }
            }
            Err(err) => {
                err_fatal(err, "poisoned rwlock on symtable");
            }
        }
    }

    fn hash_str<'a>(sym: &'a str) -> u64 {
        const PRIME: u64 = 0x100000001b3;
        let mut state: u64 = 0xCCCC_FFFF_FFFF_CCCC;

        for byte in sym.as_bytes() {
            state = state.wrapping_shl(8) ^ PRIME;
            state = state.wrapping_shl(8) ^ (*byte as u64);
        }

        state
    }
}

impl std::cmp::PartialEq for SymTable {
    fn eq(&self, other: &Self) -> bool {
        self.astid == other.astid
    }
}

impl std::cmp::Eq for SymTable {}

impl std::cmp::PartialOrd for SymTable {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.astid.partial_cmp(&other.astid)
    }
}

impl std::cmp::Ord for SymTable {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.astid.cmp(&other.astid)
    }
}

impl Deref for SymTable {
    type Target = Arc<RwLock<HashMap<u64, String>>>;

    fn deref(&self) -> &Self::Target {
        &self.table
    }
}

impl Hash for SymTable {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.astid.hash(state);
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sym {
    index: u64,
    table: SymTable,
}

impl Sym {
    pub fn parse<T: FromStr>(&self) -> Result<T, <T as FromStr>::Err> {
        let strval = self.as_string();
        strval.parse::<T>()
    }

    pub fn as_string(&self) -> String {
        match self.table.read() {
            Ok(table) => match table.get(&self.index) {
                Some(strval) => {
                    return strval.clone();
                }
                None => {
                    return String::from("unknown symbol");
                }
            },
            Err(_) => {
                return String::from("sym:[internal error]");
            }
        }
    }
}

impl Display for Sym {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.table.read() {
            Ok(table) => {
                write!(
                    f,
                    "{}",
                    table
                        .get(&self.index)
                        .unwrap_or(&String::from("unknown symbol"))
                )
            }
            Err(_) => {
                write!(f, "sym:[internal error]")
            }
        }
    }
}

impl Debug for Sym {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.table.read() {
            Ok(table) => {
                write!(
                    f,
                    "[SYM:{:?}]",
                    table
                        .get_key_value(&self.index)
                        .unwrap_or((&0, &String::from("unknown symbol")))
                )
            }
            Err(_) => {
                write!(f, "[internal error]")
            }
        }
    }
}

fn err_fatal<E: Error>(err: E, why: &'static str) -> ! {
    panic!("fatal internal error: {why},\n{err}")
}
