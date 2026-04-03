
{
    fn main() {
        10 + 10
    }

    fn foo(qq: f32) {
        let local_1 = 3.3 + qq;
        let bar = 10.0;
        let g = 9.0 + 4.0 * local_1 + 2.0 / 3.0 - 99.0 * bar;
        g * g;
    }

    fn bar(a: i32, b: i32) {
        a * a + b * b
    }
}
