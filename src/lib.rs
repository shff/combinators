#[derive(Debug, PartialEq, Clone)]
pub enum Token<'a> {
    Id(&'a str),
    Op(&'a str),
    Num(&'a str),
    Sum((Box<Token<'a>>, Box<Token<'a>>)),
    BinOp((&'a str, Box<Token<'a>>, Box<Token<'a>>)),
    Prefix((&'a str, Box<Token<'a>>)),
}

// Parsers

pub fn assignment<'a>(i: &'a str) -> ParseResult<(Token<'a>, Token<'a>)> {
    outer(w(identifier), w(token("=")), w(expression))(i)
}

// Expressions

fn expression<'a>(i: &'a str) -> ParseResult<Token<'a>> {
    sum_sub(i)
}

fn sum_sub<'a>(i: &'a str) -> ParseResult<Token<'a>> {
    infix(w(mul_div), w(any(token("+"), token("-"))), Token::BinOp)(i)
}

fn mul_div<'a>(i: &'a str) -> ParseResult<Token<'a>> {
    let ops = w(any3(token("*"), token("/"), token("%")));
    infix(w(exp), ops, Token::BinOp)(i)
}

fn exp<'a>(i: &'a str) -> ParseResult<Token<'a>> {
    infix(w(unary), w(token("**")), Token::BinOp)(i)
}

fn unary<'a>(i: &'a str) -> ParseResult<Token<'a>> {
    let prefix = |x, y, c| map(pair(x, b(w(y))), c);
    let ops = w(any3(token("+"), token("-"), token("!")));
    any(prefix(ops, primitive, Token::Prefix), primitive)(i)
}

fn primitive<'a>(i: &'a str) -> ParseResult<Token<'a>> {
    any(parens, any(number, identifier))(i)
}

// Programming Primitives

fn identifier<'a>(i: &'a str) -> ParseResult<Token<'a>> {
    map(join(pair(alphabetic, opt(alphanumeric))), Token::Id)(i)
}

fn number<'a>(i: &'a str) -> ParseResult<Token<'a>> {
    map(numeric, Token::Num)(i)
}

fn parens<'a>(i: &'a str) -> ParseResult<Token<'a>> {
    middle(w(token("(")), expression, w(token(")")))(i)
}

// Parsing Primitives

fn alphanumeric<'a>(i: &'a str) -> ParseResult<&'a str> {
    taken(|c| c.is_alphanumeric() || c == '_')(i)
}

fn alphabetic<'a>(i: &'a str) -> ParseResult<&'a str> {
    taken(|c| c.is_alphabetic())(i)
}

fn numeric<'a>(i: &'a str) -> ParseResult<&'a str> {
    taken(|c| c.is_numeric() || c == '_')(i)
}

// Combinators

pub type ParseResult<'a, T> = Result<(&'a str, T), (&'a str, ParserError<'a>)>;

#[derive(Debug, PartialEq)]
pub enum ParserError<'a> {
    Expected(&'a str),
    Take,
}

fn token<'a>(token: &'a str) -> impl Fn(&'a str) -> ParseResult<&'a str> {
    move |i| match i.starts_with(token) {
        true => Ok((&i[token.len()..], &i[..token.len()])),
        false => Err((i, ParserError::Expected(&token))),
    }
}

fn left<'a, A, B, X, Y>(a: A, b: B) -> impl Fn(&'a str) -> ParseResult<X>
where
    A: Fn(&'a str) -> ParseResult<X>,
    B: Fn(&'a str) -> ParseResult<Y>,
{
    move |i| a(i).and_then(|(i, r1)| b(i).map(|(i, _)| (i, r1)))
}

fn right<'a, A, B, X, Y>(a: A, b: B) -> impl Fn(&'a str) -> ParseResult<Y>
where
    A: Fn(&'a str) -> ParseResult<X>,
    B: Fn(&'a str) -> ParseResult<Y>,
{
    move |i| a(i).and_then(|(i, _)| b(i).map(|(i, r2)| (i, r2)))
}

fn middle<'a, A, B, C, X, Y, Z>(a: A, b: B, c: C) -> impl Fn(&'a str) -> ParseResult<Y>
where
    A: Fn(&'a str) -> ParseResult<X>,
    B: Fn(&'a str) -> ParseResult<Y>,
    C: Fn(&'a str) -> ParseResult<Z>,
{
    move |i| a(i).and_then(|(i, _)| b(i).and_then(|(i, r2)| c(i).map(|(i, _)| (i, r2))))
}

fn outer<'a, A, B, C, X, Y, Z>(a: A, b: B, c: C) -> impl Fn(&'a str) -> ParseResult<(X, Z)>
where
    A: Fn(&'a str) -> ParseResult<X>,
    B: Fn(&'a str) -> ParseResult<Y>,
    C: Fn(&'a str) -> ParseResult<Z>,
{
    move |i| a(i).and_then(|(i, x)| b(i).and_then(|(i, _)| c(i).map(|(i, z)| (i, (x, z)))))
}

fn pair<'a, A, B, X, Y>(a: A, b: B) -> impl Fn(&'a str) -> ParseResult<(X, Y)>
where
    A: Fn(&'a str) -> ParseResult<X>,
    B: Fn(&'a str) -> ParseResult<Y>,
{
    move |i| a(i).and_then(|(i, r1)| b(i).map(|(i, r2)| (i, (r1, r2))))
}

fn any<'a, A, B, X>(a: A, b: B) -> impl Fn(&'a str) -> ParseResult<X>
where
    A: Fn(&'a str) -> ParseResult<X>,
    B: Fn(&'a str) -> ParseResult<X>,
{
    move |i| a(i).or(b(i))
}

fn any3<'a, A, B, C, X>(a: A, b: B, c: C) -> impl Fn(&'a str) -> ParseResult<X>
where
    A: Fn(&'a str) -> ParseResult<X>,
    B: Fn(&'a str) -> ParseResult<X>,
    C: Fn(&'a str) -> ParseResult<X>,
{
    move |i| a(i).or(b(i)).or(c(i))
}

fn opt<'a, P, R>(p: P) -> impl Fn(&'a str) -> ParseResult<Option<R>>
where
    P: Fn(&'a str) -> ParseResult<R>,
{
    move |i| p(i).map(|(i, r)| (i, Some(r))).or(Ok((i, None)))
}

fn map<'a, P, F, A, B>(p: P, f: F) -> impl Fn(&'a str) -> ParseResult<B>
where
    P: Fn(&'a str) -> ParseResult<A>,
    F: Fn(A) -> B,
{
    move |i| p(i).map(|(i, r)| (i, f(r)))
}

fn many<'a, P, R>(p: P) -> impl Fn(&'a str) -> ParseResult<Vec<R>>
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

fn take<'a, P>(p: P) -> impl Fn(&'a str) -> ParseResult<&str>
where
    P: Copy + Fn(char) -> bool,
{
    move |i| match i.find(|c| !p(c)) {
        Some(x) => Ok((&i[x..], &i[..x])),
        None => Ok((&i[i.len()..], i)),
    }
}

fn taken<'a, P>(p: P) -> impl Fn(&'a str) -> ParseResult<&str>
where
    P: Copy + Fn(char) -> bool,
{
    move |i| match i.find(|c| !p(c)) {
        Some(0) => Err((i, ParserError::Take)),
        Some(x) => Ok((&i[x..], &i[..x])),
        None => Ok((&i[i.len()..], i)),
    }
}

fn join<'a, P, T>(p: P) -> impl Fn(&'a str) -> ParseResult<&str>
where
    P: Fn(&'a str) -> ParseResult<T>,
{
    move |i| match p(i) {
        Ok((i2, _)) => Ok((i2, &i[..(i2.as_ptr() as usize - i.as_ptr() as usize)])),
        Err(b) => Err(b),
    }
}

fn infix<'a, A, B, C, T, O>(a: A, b: B, c: C) -> impl Fn(&'a str) -> ParseResult<T>
where
    A: Fn(&'a str) -> ParseResult<T>,
    B: Fn(&'a str) -> ParseResult<O>,
    C: Fn((O, Box<T>, Box<T>)) -> T,
{
    move |i| {
        a(i).map(|(mut i, mut curr)| {
            while let Ok((next_input, (op, next))) = pair(&b, &a)(i) {
                i = next_input;
                curr = c((op, Box::new(curr), Box::new(next)))
            }
            (i, curr)
        })
    }
}

pub fn b<'a, F, T>(f: F) -> impl Fn(&'a str) -> ParseResult<Box<T>>
where
    F: Fn(&'a str) -> ParseResult<T>,
{
    map(f, Box::new)
}

fn w<'a, F, T>(f: F) -> impl Fn(&'a str) -> ParseResult<T>
where
    F: Fn(&'a str) -> ParseResult<T>,
{
    right(take(|c: char| c.is_whitespace()), f)
}

#[test]
fn test_parser() {
    let parser = expression;
    assert_eq!(parser("123 x"), Ok((" x", Token::Num("123"))));
    assert_eq!(parser("a("), Ok(("(", Token::Id("a"))));
    assert_eq!(parser("(a)"), Ok(("", Token::Id("a"))));
    let n = Box::new(Token::Num("1"));
    assert_eq!(
        parser("( 1 + 1 )"),
        Ok(("", Token::BinOp(("+", n.clone(), n.clone()))))
    );
    assert_eq!(
        parser("( 1 - 1 )"),
        Ok(("", Token::BinOp(("-", n.clone(), n.clone()))))
    );
    assert_eq!(
        parser("( 1 * 1 )"),
        Ok(("", Token::BinOp(("*", n.clone(), n.clone()))))
    );
    assert_eq!(
        parser("( 1 / 1 )"),
        Ok(("", Token::BinOp(("/", n.clone(), n.clone()))))
    );
    assert_eq!(
        parser("( 1 % 1 )"),
        Ok(("", Token::BinOp(("%", n.clone(), n.clone()))))
    );
    assert_eq!(
        parser("( 1 ** 1 )"),
        Ok(("", Token::BinOp(("**", n.clone(), n.clone()))))
    );
    assert_eq!(parser("( ! 1 )"), Ok(("", Token::Prefix(("!", n.clone())))));
    assert_eq!(parser("( - 1 )"), Ok(("", Token::Prefix(("-", n.clone())))));
    assert_eq!(parser("( + 1 )"), Ok(("", Token::Prefix(("+", n.clone())))));

    let parser = identifier;
    assert_eq!(parser("abc_123*"), Ok(("*", Token::Id("abc_123"))));
    assert_eq!(parser("a("), Ok(("(", Token::Id("a"))));
    assert_eq!(parser("2abc123"), Err(("2abc123", ParserError::Take)));
    assert_eq!(parser("*bcda"), Err(("*bcda", ParserError::Take)));

    let parser = number;
    assert_eq!(parser("123 x"), Ok((" x", Token::Num("123"))));
    assert_eq!(parser("x 123 x"), Err(("x 123 x", ParserError::Take)));

    let parser = assignment;
    assert_eq!(parser("x = y"), Ok(("", (Token::Id("x"), Token::Id("y")))));
    assert_eq!(parser("a=c"), Ok(("", (Token::Id("a"), Token::Id("c")))));
    assert_eq!(parser("a=22"), Ok(("", (Token::Id("a"), Token::Num("22")))));
    assert_eq!(parser("22=a"), Err(("22=a", ParserError::Take)));
}

#[test]
fn test_combinators() {
    let parser = token("a");
    assert_eq!(parser("ab"), Ok(("b", "a")));
    assert_eq!(parser("bb"), Err(("bb", ParserError::Expected("a"))));

    let parser = left(token("a"), token("b"));
    assert_eq!(parser("ab"), Ok(("", "a")));
    assert_eq!(parser("bb"), Err(("bb", ParserError::Expected("a"))));

    let parser = right(token("a"), token("b"));
    assert_eq!(parser("ab"), Ok(("", "b")));
    assert_eq!(parser("aa"), Err(("a", ParserError::Expected("b"))));

    let parser = middle(token("a"), token("b"), token("c"));
    assert_eq!(parser("abc"), Ok(("", "b")));
    assert_eq!(parser("b"), Err(("b", ParserError::Expected("a"))));

    let parser = outer(token("a"), token("b"), token("c"));
    assert_eq!(parser("abc"), Ok(("", ("a", "c"))));
    assert_eq!(parser("bca"), Err(("bca", ParserError::Expected("a"))));

    let parser = pair(token("a"), token("b"));
    assert_eq!(parser("ab"), Ok(("", ("a", "b"))));
    assert_eq!(parser("aa"), Err(("a", ParserError::Expected("b"))));

    let parser = any(token("aa"), token("bb"));
    assert_eq!(parser("aabb"), Ok(("bb", "aa")));
    assert_eq!(parser("bbaa"), Ok(("aa", "bb")));
    assert_eq!(parser("ccbb"), Err(("ccbb", ParserError::Expected("bb"))));

    let parser = opt(token("ab"));
    assert_eq!(parser("ab"), Ok(("", Some("ab"))));
    assert_eq!(parser("ba"), Ok(("ba", None)));

    let parser = map(token("1"), |s| s.parse::<i32>().unwrap());
    assert_eq!(parser("1"), Ok(("", 1)));
    assert_eq!(parser("2"), Err(("2", ParserError::Expected("1"))));

    let parser = many(token("a"));
    assert_eq!(parser("aaaa"), Ok(("", vec!["a", "a", "a", "a"])));
    assert_eq!(parser("baaa"), Ok(("baaa", vec![])));

    let parser = take(|c| c == 'a');
    assert_eq!(parser("aaaa"), Ok(("", "aaaa")));
    assert_eq!(parser("baaa"), Ok(("baaa", "")));

    let parser = taken(|c| c == 'a');
    assert_eq!(parser("aaaa"), Ok(("", "aaaa")));
    assert_eq!(parser("baaa"), Err(("baaa", ParserError::Take)));

    let parser = join(pair(token("a"), many(token("b"))));
    assert_eq!(parser("abbb"), Ok(("", "abbb")));
    assert_eq!(parser("ab"), Ok(("", "ab")));
    assert_eq!(parser("a"), Ok(("", "a")));
    assert_eq!(parser("baa"), Err(("baa", ParserError::Expected("a"))));

    #[derive(Debug, PartialEq, Clone)]
    pub enum T<'a> {
        Str(&'a str),
        Pair((&'a str, Box<T<'a>>, Box<T<'a>>)),
    }
    let a = Box::new(T::Str("a"));

    let parser = infix(map(token("a"), T::Str), token("b"), T::Pair);
    assert_eq!(parser("ab"), Ok(("b", T::Str("a"))));
    assert_eq!(parser("abab"), Ok(("b", T::Pair(("b", a.clone(), a)))));

    let parser = w(token("a"));
    assert_eq!(parser("   a"), Ok(("", "a")));
    assert_eq!(parser("a   "), Ok(("   ", "a")));
}
