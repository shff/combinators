pub type ParseResult<'a, T> = Result<(&'a str, T), (&'a str, ParserError<'a>)>;

#[derive(Debug, PartialEq)]
pub enum ParserError<'a> {
    Expected(&'a str),
}

pub fn t<'a>(token: &'a str) -> impl Fn(&'a str) -> ParseResult<&'a str> {
    move |i| match i.starts_with(token) {
        true => Ok((&i[token.len()..], &i[..token.len()])),
        false => Err((i, ParserError::Expected(&token))),
    }
}

pub fn l<'a, A, B, X, Y>(a: A, b: B) -> impl Fn(&'a str) -> ParseResult<X>
where
    A: Fn(&'a str) -> ParseResult<X>,
    B: Fn(&'a str) -> ParseResult<Y>,
{
    move |i| a(i).and_then(|(i, r1)| b(i).map(|(i, _)| (i, r1)))
}

pub fn r<'a, A, B, X, Y>(a: A, b: B) -> impl Fn(&'a str) -> ParseResult<Y>
where
    A: Fn(&'a str) -> ParseResult<X>,
    B: Fn(&'a str) -> ParseResult<Y>,
{
    move |i| a(i).and_then(|(i, _)| b(i).map(|(i, r2)| (i, r2)))
}

pub fn m<'a, A, B, C, X, Y, Z>(a: A, b: B, c: C) -> impl Fn(&'a str) -> ParseResult<Y>
where
    A: Fn(&'a str) -> ParseResult<X>,
    B: Fn(&'a str) -> ParseResult<Y>,
    C: Fn(&'a str) -> ParseResult<Z>,
{
    move |i| a(i).and_then(|(i, _)| b(i).and_then(|(i, r2)| c(i).map(|(i, _)| (i, r2))))
}

pub fn o<'a, A, B, C, X, Y, Z>(a: A, b: B, c: C) -> impl Fn(&'a str) -> ParseResult<(X, Z)>
where
    A: Fn(&'a str) -> ParseResult<X>,
    B: Fn(&'a str) -> ParseResult<Y>,
    C: Fn(&'a str) -> ParseResult<Z>,
{
    move |i| a(i).and_then(|(i, x)| b(i).and_then(|(i, _)| c(i).map(|(i, z)| (i, (x, z)))))
}

pub fn b<'a, A, B, X, Y>(a: A, b: B) -> impl Fn(&'a str) -> ParseResult<(X, Y)>
where
    A: Fn(&'a str) -> ParseResult<X>,
    B: Fn(&'a str) -> ParseResult<Y>,
{
    move |i| a(i).and_then(|(i, r1)| b(i).map(|(i, r2)| (i, (r1, r2))))
}

pub fn opt<'a, P, R>(p: P) -> impl Fn(&'a str) -> ParseResult<Option<R>>
where
    P: Fn(&'a str) -> ParseResult<R>,
{
    move |i| p(i).map(|(i, r)| (i, Some(r))).or(Ok((i, None)))
}

pub fn map<'a, P, F, A, B>(p: P, f: F) -> impl Fn(&'a str) -> ParseResult<B>
where
    P: Fn(&'a str) -> ParseResult<A>,
    F: Fn(A) -> B,
{
    move |i| p(i).map(|(i, r)| (i, f(r)))
}

pub fn n<'a, P, R>(p: P) -> impl Fn(&'a str) -> ParseResult<Vec<R>>
where
    P: Fn(&'a str) -> ParseResult<R>,
{
    move |mut i| {
        let mut r = Vec::new();
        while let Ok((next_input, next_item)) = p(i) {
            i = next_input;
            r.push(next_item);
        }
        Ok((i, r))
    }
}

pub fn take<'a, P>(p: P) -> impl Fn(&'a str) -> ParseResult<&str>
where
    P: Copy + Fn(char) -> bool,
{
    move |i| match i.find(|c| !p(c)) {
        Some(x) => Ok((&i[x..], &i[..x])),
        None => Ok((&i[i.len()..], i))
    }
}

pub fn j<'a, P, T>(p: P) -> impl Fn(&'a str) -> ParseResult<&str>
where
    P: Fn(&'a str) -> ParseResult<T>
{
    move |i| match p(i) {
        Ok((i2, _)) => Ok((i2, &i[..(i2.as_ptr() as usize - i.as_ptr() as usize)])),
        Err(b) => Err(b)
    }
}

pub fn w<'a, F, T>(f: F) -> impl Fn(&'a str) -> ParseResult<T>
where
    F: Fn(&'a str) -> ParseResult<T>,
{
    r(take(|c: char| c.is_whitespace()), f)
}

#[test]
fn test_combinators() {
    let parser = t("a");
    assert_eq!(parser("ab"), Ok(("b", "a")));
    assert_eq!(parser("bb"), Err(("bb", ParserError::Expected("a"))));

    let parser = l(t("a"), t("b"));
    assert_eq!(parser("ab"), Ok(("", "a")));
    assert_eq!(parser("bb"), Err(("bb", ParserError::Expected("a"))));

    let parser = r(t("a"), t("b"));
    assert_eq!(parser("ab"), Ok(("", "b")));
    assert_eq!(parser("aa"), Err(("a", ParserError::Expected("b"))));

    let parser = m(t("a"), t("b"), t("c"));
    assert_eq!(parser("abc"), Ok(("", "b")));
    assert_eq!(parser("b"), Err(("b", ParserError::Expected("a"))));

    let parser = o(t("a"), t("b"), t("c"));
    assert_eq!(parser("abc"), Ok(("", ("a", "c"))));
    assert_eq!(parser("bca"), Err(("bca", ParserError::Expected("a"))));

    let parser = b(t("a"), t("b"));
    assert_eq!(parser("ab"), Ok(("", ("a", "b"))));
    assert_eq!(parser("aa"), Err(("a", ParserError::Expected("b"))));

    let parser = opt(t("ab"));
    assert_eq!(parser("ab"), Ok(("", Some("ab"))));
    assert_eq!(parser("ba"), Ok(("ba", None)));

    let parser = map(t("1"), |s| s.parse::<i32>().unwrap());
    assert_eq!(parser("1"), Ok(("", 1)));
    assert_eq!(parser("2"), Err(("2", ParserError::Expected("1"))));

    let parser = n(t("a"));
    assert_eq!(parser("aaaa"), Ok(("", vec!["a", "a", "a", "a"])));
    assert_eq!(parser("baaa"), Ok(("baaa", vec![])));

    let parser = take(|c| c == 'a');
    assert_eq!(parser("aaaa"), Ok(("", "aaaa")));
    assert_eq!(parser("baaa"), Ok(("baaa", "")));

    let parser = j(b(t("a"), n(t("b"))));
    assert_eq!(parser("abbb"), Ok(("", "abbb")));
    assert_eq!(parser("ab"), Ok(("", "ab")));
    assert_eq!(parser("a"), Ok(("", "a")));
    assert_eq!(parser("baa"), Err(("baa", ParserError::Expected("a"))));

    let parser = w(t("a"));
    assert_eq!(parser("   a"), Ok(("", "a")));
    assert_eq!(parser("a   "), Ok(("   ", "a")));
}
