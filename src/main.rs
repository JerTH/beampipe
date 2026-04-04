use std::fs;
use std::path::Path;
use std::process;

use beampipe::emit::Emit;
use beampipe::error::format_error;
use beampipe::eval::Eval;
use beampipe::parse::Parse;

struct Args {
    file: String,
    eval: bool,
    ir: bool,
    bytecode: bool,
    ast: bool,
    output: Option<String>,
}

fn print_usage() {
    eprintln!(
        "USAGE: beampipe <FILE> [OPTIONS]

OPTIONS:
    --eval          Evaluate the program and print the result to stdout
    --ir            Produce a <stem>.ir file containing the IR listing
    --bytecode      Produce a <stem>.bc file containing bytecode
    --ast           Produce a <stem>.ast file containing the AST
    -o, --output    Output name stem (default: input filename without extension)
    -h, --help      Print this help information

With no flags the default action is to compile, producing both <stem>.ir
and <stem>.bc files."
    );
}

fn parse_args() -> Result<Args, String> {
    let mut args_iter = std::env::args().skip(1);
    let mut file = None;
    let mut eval = false;
    let mut ir = false;
    let mut bytecode = false;
    let mut ast = false;
    let mut output = None;

    while let Some(arg) = args_iter.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                process::exit(0);
            }
            "--eval" => eval = true,
            "--ir" => ir = true,
            "--bytecode" => bytecode = true,
            "--ast" => ast = true,
            "-o" | "--output" => {
                output = Some(
                    args_iter
                        .next()
                        .ok_or_else(|| format!("{} requires a value", arg))?,
                );
            }
            s if s.starts_with('-') => {
                return Err(format!("unknown option: {}", s));
            }
            _ => {
                if file.is_some() {
                    return Err("multiple input files are not supported".into());
                }
                file = Some(arg);
            }
        }
    }

    let file = file.ok_or("no input file provided")?;
    Ok(Args { file, eval, ir, bytecode, ast, output })
}

fn extract_stem(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output")
        .to_string()
}

fn write_output(path: &str, contents: &str) {
    if let Err(e) = fs::write(path, contents) {
        eprintln!("error: failed to write {}: {}", path, e);
        process::exit(1);
    }
    eprintln!("wrote {}", path);
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(msg) => {
            eprintln!("error: {}", msg);
            print_usage();
            process::exit(1);
        }
    };

    let source = match fs::read_to_string(&args.file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {}: {}", args.file, e);
            process::exit(1);
        }
    };

    let expr = match Parse::parse(&source) {
        Ok(expr) => expr,
        Err(e) => {
            eprintln!("error: parse failed\n{}", e);
            process::exit(1);
        }
    };

    let stem = args.output.unwrap_or_else(|| extract_stem(&args.file));

    if args.eval {
        match Eval::eval(&expr) {
            Ok(value) => println!("{}", value),
            Err(e) => {
                eprintln!("{}", format_error(&e, &source));
                process::exit(1);
            }
        }
        return;
    }

    let (do_ir, do_bc, do_ast) = if !args.ir && !args.bytecode && !args.ast {
        (true, true, false)
    } else {
        (args.ir, args.bytecode, args.ast)
    };

    if do_ir || do_bc {
        let ir_code = match Emit::emit(&expr) {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{}", format_error(&e, &source));
                process::exit(1);
            }
        };
        let ir_text = format!("{}", ir_code);
        if do_ir {
            write_output(&format!("{}.ir", stem), &ir_text);
        }
        if do_bc {
            write_output(&format!("{}.bc", stem), &ir_text);
        }
    }

    if do_ast {
        let ast_text = format!("{:#?}", expr);
        write_output(&format!("{}.ast", stem), &ast_text);
    }
}
