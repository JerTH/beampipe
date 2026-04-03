use crate::ast::Span;
use crate::token::{self, Token, TokenK};

pub fn tokenize(source: &str) -> Vec<Token> {
    let mut lexer = Lexer {
        source: source.to_string(),
        tokens: Vec::new(),
    };
    lexer.peel_tokens();
    lexer.tokens
}

struct Lexer {
    source: String,
    tokens: Vec<Token>,
}

impl Lexer {
    fn peel_tokens(&mut self) {
        let source = self.source.clone();
        let chars: Vec<char> = source.chars().collect();
        let mut buff: Vec<char> = Vec::new();

        let iterator = chars.iter().enumerate();

        for (curs, char) in iterator {
            if char.is_alphanumeric() | char.is_whitespace() | !char.is_multi_part() {
                self.match_key(curs, &mut buff);
            }

            if !char.is_numeric() & (*char != '.') {
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
        if !self.match_key(chars.len(), &mut buff)
            && !self.match_ident(chars.len(), &mut buff) {
                self.match_numeric(chars.len(), &mut buff);
            }
    }

    fn match_numeric(&mut self, curs: usize, buff: &mut Vec<char>) -> bool {
        if buff.is_empty() {
            return false;
        }
        let string_repr = String::from_iter(buff.iter());

        let tspan = Span {
            bgn: curs - buff.len(),
            end: curs - 1,
        };

        let mut token = None;

        if string_repr.parse::<i64>().is_ok() {
            token = Some(Token::new(tspan, TokenK::LitInt))
        };

        if token.is_none() && string_repr.parse::<f64>().is_ok() {
            token = Some(Token::new(tspan, TokenK::LitFloat))
        }

        if let Some(token) = token {
            self.tokens.push(token);
            buff.clear();
            true
        } else {
            false
        }
    }

    fn match_key(&mut self, curs: usize, buff: &mut Vec<char>) -> bool {
        if buff.is_empty() {
            return false;
        }

        let string_repr = String::from_iter(buff.iter());

        let tspan = Span {
            bgn: curs - buff.len(),
            end: curs - 1,
        };

        // Handle multi-character punctuation
        let mut token = match buff.first() {
            Some('-') | Some('=') | Some(':') => match buff.get(1) {
                Some('>') => Some(Token::new(tspan, TokenK::Arrow)),
                Some('=') => Some(Token::new(tspan, TokenK::EqEq)),
                _ => None,
            },
            Some('>') => Some(Token::new(tspan, TokenK::OpGreater)),
            Some('<') => Some(Token::new(tspan, TokenK::OpLess)),
            _ => None,
        };

        if token.is_none() {
            token = match string_repr.as_str() {
                token::TOK_IF => Some(Token::new(tspan, TokenK::KeyIf)),
                token::TOK_ELSE => Some(Token::new(tspan, TokenK::KeyElse)),
                token::TOK_FN => Some(Token::new(tspan, TokenK::KeyFn)),
                token::TOK_LET => Some(Token::new(tspan, TokenK::KeyLet)),
                token::TOK_MUT => Some(Token::new(tspan, TokenK::KeyMut)),
                token::TOK_REF => Some(Token::new(tspan, TokenK::KeyRef)),
                token::TOK_WHILE => Some(Token::new(tspan, TokenK::KeyWhile)),
                token::TOK_LOOP => Some(Token::new(tspan, TokenK::KeyLoop)),
                token::TOK_RET => Some(Token::new(tspan, TokenK::KeyReturn)),
                token::TOK_TRUE => Some(Token::new(tspan, TokenK::LitBool)),
                token::TOK_FALSE => Some(Token::new(tspan, TokenK::LitBool)),

                token::TOK_USIZE => Some(Token::new(tspan, TokenK::Ty)),
                token::TOK_ISIZE => Some(Token::new(tspan, TokenK::Ty)),
                token::TOK_I32 => Some(Token::new(tspan, TokenK::Ty)),
                token::TOK_I64 => Some(Token::new(tspan, TokenK::Ty)),
                token::TOK_U32 => Some(Token::new(tspan, TokenK::Ty)),
                token::TOK_U64 => Some(Token::new(tspan, TokenK::Ty)),
                token::TOK_F32 => Some(Token::new(tspan, TokenK::Ty)),
                token::TOK_F64 => Some(Token::new(tspan, TokenK::Ty)),

                token::TOK_COLON => Some(Token::new(tspan, TokenK::Colon)),
                token::TOK_SEMI => Some(Token::new(tspan, TokenK::Semi)),
                token::TOK_COMMA => Some(Token::new(tspan, TokenK::Comma)),
                token::TOK_DOT => Some(Token::new(tspan, TokenK::Dot)),
                token::TOK_L_BRACE => Some(Token::new(tspan, TokenK::LBrace)),
                token::TOK_R_BRACE => Some(Token::new(tspan, TokenK::RBrace)),
                token::TOK_L_PAREN => Some(Token::new(tspan, TokenK::LParen)),
                token::TOK_R_PAREN => Some(Token::new(tspan, TokenK::RParen)),
                token::TOK_L_BRACK => Some(Token::new(tspan, TokenK::LBrack)),
                token::TOK_R_BRACK => Some(Token::new(tspan, TokenK::RBrack)),

                token::TOK_PLUS => Some(Token::new(tspan, TokenK::OpAdd)),
                token::TOK_MINUS => Some(Token::new(tspan, TokenK::OpSub)),
                token::TOK_STAR => Some(Token::new(tspan, TokenK::OpMul)),
                token::TOK_SLASH => Some(Token::new(tspan, TokenK::OpDiv)),
                token::TOK_EQ => Some(Token::new(tspan, TokenK::OpEq)),

                _ => None,
            };
        }

        if let Some(token) = token {
            self.tokens.push(token);
            buff.clear();
            true
        } else {
            false
        }
    }

    fn match_ident(&mut self, curs: usize, buff: &mut Vec<char>) -> bool {
        if buff.is_empty() {
            return false;
        }

        let tspan = Span {
            bgn: curs - buff.len(),
            end: curs - 1,
        };

        let matches = match buff[0] {
            c if c.is_alphabetic() | (c == '_') => {
                if buff
                    .iter()
                    .skip(1)
                    .all(|c| c.is_alphanumeric() | (*c == '_'))
                {
                    Some(Token::new(tspan, TokenK::LitIdent))
                } else {
                    None
                }
            }
            _ => None,
        };
        if let Some(token) = matches {
            self.tokens.push(token);
            buff.clear();
            true
        } else {
            false
        }
    }
}

trait CharImplExt {
    fn is_multi_part(&self) -> bool;
}

impl CharImplExt for char {
    fn is_multi_part(&self) -> bool {
        matches!(self, '>' | '=')
    }
}
