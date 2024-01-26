

fn foo(a: usize, b: usize) -> usize {
    if a > b {
        b
    } else {
        a + b
    }
}

fn bar(was: bool, were: bool) -> usize {
    if was {
        if were {
            42
        } else {
            1234
        }
    }
    1337
}

fn main() {
    bar(true, false) + foo(1, 3)
}
