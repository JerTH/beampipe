//! Macros

pub(crate) static DBG_INDENT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
macro_rules! dbg_print {
    (push, $($t:tt)*) => {
        {
            crate::macros::DBG_INDENT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        dbg_print_color!($($t)*)
    };
    
    (pop, $($t:tt)*) => {
        {
            crate::macros::DBG_INDENT.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
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
            //print!("{}", "  ".repeat(crate::macros::DBG_INDENT.load(std::sync::atomic::Ordering::SeqCst)));
            //print!($($t)*);
        }
    };
}