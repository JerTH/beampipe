fn foo(a: usize, b: usize) -> usize {
    if bar(a > b) {
        b
    } else {
        a
    }
}

fn bar(maybe_yes: bool) -> bool {
    let maybe_no = !maybe_yes;
    return maybe_no
}
