

fn foo(a: usize, b: usize) -> usize {
    if a > b {
        b
    } else {
        a
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
