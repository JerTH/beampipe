use std::{ops::{Deref, DerefMut, Bound, RangeBounds}, collections::HashMap, sync::{Arc, RwLock, atomic::{AtomicUsize, Ordering}}, error::Error, fmt::{Debug, Display}, str::FromStr, hash::Hash};

type PtrType<T> = Box<T>;

#[derive(Debug, Clone)]
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

macro_rules! p {
    ($e:expr) => {
        Ptr::new($e)
    }
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

#[derive(Debug, Clone)]
pub struct Path     { pub astid: AstId, pub list: Vec<Ident>, pub span: Span, }

impl Into<String> for &Ptr<Path> {
    fn into(self) -> String {
        self.list.iter().fold(String::new(), |acc, x| {
            format!("{}::{}", acc, x)
        })
    }
}

#[derive(Debug, Clone)]
pub struct Ty       { pub astid: AstId, pub kind: TyK, pub span: Span, }

#[derive(Debug, Clone)]
pub struct Local    { pub astid: AstId, pub ident: Ident, pub kind: LocalK, pub ty: Ptr<Ty>, pub span: Span, }

#[derive(Debug, Clone)]
pub struct Blk      { pub astid: AstId, pub list: Vec<Expr>, pub span: Span, }

#[derive(Debug, Clone)]
pub struct Item     { pub astid: AstId, pub ident: Ident, }

#[derive(Debug, Clone)]
pub struct Lit      { pub symbol: Sym, pub kind: LitK, }

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ident    { pub name: Sym, pub span: Span, }

impl Display for Ident {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name.as_string())
    }
}

#[derive(Debug, Clone)]
pub struct BinOp    { pub span: Span, pub kind: BinOpK, }

#[derive(Debug, Clone)]
pub struct Expr     { pub astid: AstId, pub kind: ExprK, pub span: Span, }

impl Expr {
    pub fn new(span: Span, kind: ExprK) -> Self {
        Expr {
            astid: astid!(),
            kind,
            span,
        }
    }
}

// Kinds
#[derive(Debug, Clone)]
pub enum TyK        { Array(Ptr<Ty>), Path(Ptr<Path>), }

#[derive(Debug, Clone)]
pub enum LocalK     { Decl, Init(Ptr<Expr>), }

#[derive(Debug, Clone)]
pub enum ItemK      {  }

#[derive(Debug, Clone)]
pub enum LitK       { Bool, Int, Float, }

#[derive(Debug, Clone)]
pub enum BinOpK     { Add, Sub, Div, Mul, }

#[derive(Debug, Clone)]
pub enum ExprK {
    Empty,
    Local(Ptr<Local>),
    Item(Ptr<Item>),
    Lit(Ptr<Lit>),
    Blk(Ptr<Blk>),
    Assign(Ptr<Expr>, Ptr<Expr>),
    BinOp(BinOp, Ptr<Expr>, Ptr<Expr>),
    AssignOp(Ptr<BinOp>, Ptr<Expr>, Ptr<Expr>),
    Path(Ptr<Path>),
}

impl ExprK {
    pub fn binop(span: Span, kind: BinOpK) -> Self {
        ExprK::BinOp(
            BinOp { span, kind },
            p!(Expr::new(span, ExprK::Empty)),
            p!(Expr::new(span, ExprK::Empty))
        )
    }
}

#[allow(dead_code)]
fn test_tree() -> Expr {
    let mut syms = SymTable::new();

    Expr {
        astid: astid!(),
        kind: ExprK::Blk(p!(Blk {
            astid: astid!(),
            list: vec![
                Expr {
                    astid: astid!(),
                    kind: ExprK::Assign(
                        p!(Expr {
                            astid: astid!(),
                            kind: ExprK::Path(p!(Path {
                                astid: astid!(),
                                list: vec![
                                    Ident {
                                        name: syms.make("my_var"),
                                        span: Span::none(),
                                    }
                                ],
                                span: Span::none(),
                            })),
                            span: Span::none(),
                        }),
                        p!(Expr {
                            astid: astid!(),
                            kind: ExprK::Lit(p!(Lit {
                                symbol: syms.make("69"),
                                kind: LitK::Int
                            })),
                            span: Span::none(),
                        }),
                    ),
                    span: Span::none(),
                },

                Expr {
                    astid:astid!(), 
                    kind: ExprK::Assign(
                        p!(Expr {
                            astid: astid!(),
                            kind: ExprK::Path(p!(Path {
                                astid: astid!(),
                                list: vec![Ident {
                                    name: syms.make("my_var"),
                                    span: Span::none(),
                                }],
                                span: Span::none(),
                            })),
                            span: Span::none(),
                        }),
                        p!(Expr {
                            astid: astid!(),
                            kind: ExprK::BinOp(
                                BinOp {
                                    span: Span::none(),
                                    kind: BinOpK::Add
                                },
                                p!(Expr {
                                    astid: astid!(),
                                    kind: ExprK::Lit(p!(Lit {
                                        symbol: syms.make("1"),
                                        kind: LitK::Int,
                                    })),
                                    span: Span::none(),
                                }),
                                p!(Expr {
                                    astid: astid!(),
                                    kind: ExprK::Lit(p!(Lit {
                                        symbol: syms.make("2"),
                                        kind: LitK::Int,
                                    })),
                                    span: Span::none(),
                                }),
                            ),
                            span: Span::none(),
                        })
                    ), 
                    span: Span::none() 
                },

                Expr {
                    astid: astid!(),
                    kind: ExprK::Path(p!(Path {
                        astid: astid!(),
                        list: vec![Ident {
                            name: syms.make("my_var"),
                            span: Span::none(),
                        }],
                        span: Span::none(),
                    })),
                    span: Span::none(),
                }
            ],
            span: Span::none(), 
        })),
        span: Span::none(),
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

                guard.insert(key, string.into());
                Sym { index: key, table: self.clone() }
            },
            Err(err) => {
                err_fatal(err, "poisoned rwlock on symtable");
            },
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

impl std::cmp::Eq for SymTable { }

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
            Ok(table) => {
                match table.get(&self.index) {
                    Some(strval) => {
                        return strval.clone();
                    },
                    None => {
                        return String::from("unknown symbol");
                    },
                }
            },
            Err(_) => {
                return String::from("sym:[internal error]");
            },
        }
    }
}

impl Display for Sym {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.table.read() {
            Ok(table) => {
                write!(f, "{}", table.get(&self.index).unwrap_or(&String::from("unknown symbol")))
            },
            Err(_) => {
                write!(f, "sym:[internal error]")
            },
        }
    }
}

impl Debug for Sym {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.table.read() {
            Ok(table) => {
                write!(f, "[SYM:{}]", table.get(&self.index).unwrap_or(&String::from("unknown symbol")))
            },
            Err(_) => {
                write!(f, "[internal error]")
            },
        }
    }
}

fn err_fatal<E: Error>(err: E, why: &'static str) -> ! {
    panic!("fatal internal error: {why},\n{err}")
}

#[cfg(test)]
mod test {
    use crate::codegen::{Eval, Emit};

    use super::*;

    #[test]
    fn make_test_tree() {
        let tree = test_tree();
        let code = Emit::emit(&tree);
        let result = Eval::eval(&tree);
        println!("{result}");
        println!("{code}");
    }
}
