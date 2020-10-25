use std::collections::{HashMap, HashSet};

pub enum Syntax<'a> {
    Lambda(&'a str, Box<Syntax<'a>>),
    Identifier(&'a str),
    Apply(Box<Syntax<'a>>, Box<Syntax<'a>>),
    Let(&'a str, Box<Syntax<'a>>, Box<Syntax<'a>>),
    Letrec(&'a str, Box<Syntax<'a>>, Box<Syntax<'a>>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type<'a> {
    Variable(Option<usize>),
    Operator(&'a str, Vec<usize>),
}

pub fn new_variable(a: &mut Vec<Type>) -> usize {
    a.push(Type::Variable(None));
    a.len() - 1
}

pub fn new_operator<'a>(a: &mut Vec<Type<'a>>, name: &'a str, types: &[usize]) -> usize {
    a.push(Type::Operator(name, types.to_vec()));
    a.len() - 1
}

pub fn analyse<'a>(
    a: &'a mut Vec<Type>,
    node: &Syntax,
    env: &HashMap<&'a str, usize>,
    non_generic: &HashSet<usize>,
) -> usize {
    match node {
        &Syntax::Identifier(ref name) => {
            if let Some(value) = env.get(name) {
                let mat = non_generic.iter().cloned().collect::<Vec<_>>();
                fresh(a, *value, &mut HashMap::new(), &mat)
            } else if name.parse::<isize>().is_ok() {
                0
            } else {
                panic!("Undefined symbol {:?}", name);
            }
        }
        &Syntax::Apply(ref func, ref arg) => {
            let fun_type = analyse(a, func, env, non_generic);
            let arg_type = analyse(a, arg, env, non_generic);
            let result_type = new_variable(a);
            let first = new_operator(a, "->", &[arg_type, result_type.clone()]);
            unify(a, first, fun_type);
            result_type
        }
        &Syntax::Lambda(ref v, ref body) => {
            let arg_type = new_variable(a);
            let mut new_env = env.clone();
            new_env.insert(v, arg_type);
            let mut new_non_generic = non_generic.clone();
            new_non_generic.insert(arg_type.clone());
            let result_type = analyse(a, body, &new_env, &new_non_generic);
            new_operator(a, "->", &[arg_type, result_type])
        }
        &Syntax::Let(ref v, ref defn, ref body) => {
            let defn_type = analyse(a, defn, env, non_generic);
            let mut new_env = env.clone();
            new_env.insert(v, defn_type);
            analyse(a, body, &new_env, non_generic)
        }
        &Syntax::Letrec(ref v, ref defn, ref body) => {
            let new_type = new_variable(a);
            let mut new_env = env.clone();
            new_env.insert(v, new_type.clone());
            let mut new_non_generic = non_generic.clone();
            new_non_generic.insert(new_type.clone());
            let defn_type = analyse(a, defn, &new_env, &new_non_generic);
            unify(a, new_type, defn_type);
            analyse(a, body, &new_env, non_generic)
        }
    }
}

fn fresh(
    a: &mut Vec<Type>,
    t: usize,
    mappings: &mut HashMap<usize, usize>,
    non_generic: &[usize],
) -> usize {
    let p = prune(a, t);
    match a.get(p).unwrap().clone() {
        Type::Variable(_) if non_generic.iter().any(|t| occurs_in_type(a, p, *t)) => p,
        Type::Variable(_) => mappings.entry(p).or_insert(new_variable(a)).clone(),
        Type::Operator(ref name, ref types) => {
            let b = types
                .iter()
                .map(|x| fresh(a, *x, mappings, non_generic))
                .collect::<Vec<_>>();
            new_operator(a, name, &b)
        }
    }
}

fn unify<'a>(alloc: &'a mut Vec<Type>, t1: usize, t2: usize) {
    let a = prune(alloc, t1);
    let b = prune(alloc, t2);
    match (alloc.get(a).unwrap().clone(), alloc.get(b).unwrap().clone()) {
        (Type::Variable(_), _) if a == b => {}
        (Type::Variable(_), _) if occurs_in_type(alloc, a, b) => panic!("recursive unification"),
        (Type::Variable(_), _) => match alloc.get_mut(a) {
            Some(&mut Type::Variable(ref mut inst)) => {
                *inst = Some(b);
            }
            _ => unimplemented!(),
        },
        (Type::Operator(_, _), Type::Variable(_)) => unify(alloc, b, a),
        (Type::Operator(ref a_name, ref a_types), Type::Operator(ref b_name, ref b_types)) => {
            if a_name != b_name || a_types.len() != b_types.len() {
                panic!("type mismatch");
            }
            for (p, q) in a_types.iter().zip(b_types.iter()) {
                unify(alloc, *p, *q);
            }
        }
    }
}

fn prune(a: &mut Vec<Type>, t: usize) -> usize {
    match a.get_mut(t).unwrap().clone() {
        Type::Variable(Some(ref mut instance)) => {
            let value = prune(a, *instance);
            *instance = value;
            value
        }
        _ => t,
    }
}

fn occurs_in_type(a: &mut Vec<Type>, v: usize, type2: usize) -> bool {
    let pruned_type2 = prune(a, type2);
    if pruned_type2 == v {
        return true;
    }
    match a.get(pruned_type2) {
        Some(Type::Operator(_, ref types)) => {
            types.clone().iter().any(|t2| occurs_in_type(a, v, *t2))
        }
        None => unreachable!(),
        _ => false,
    }
}

#[cfg(test)]
mod test {
    use super::*;

    impl Type<'_> {
        fn id(&self, a: &Vec<Type>) -> usize {
            a.iter()
                .position(|r| self as *const _ == r as *const _)
                .unwrap()
        }

        fn as_string(&self, a: &Vec<Type>, namer: &mut Namer) -> String {
            match self {
                &Type::Variable(Some(inst)) => a[inst].as_string(a, namer),
                &Type::Variable(_) => namer.name(self.id(a)),
                &Type::Operator(ref name, ref types) => match types.len() {
                    0 => name.to_string(),
                    2 => {
                        let l = a[types[0]].as_string(a, namer);
                        let r = a[types[1]].as_string(a, namer);
                        format!("({} {} {})", l, name, r)
                    }
                    _ => {
                        let mut coll = vec![];
                        for v in types {
                            coll.push(a[*v].as_string(a, namer));
                        }
                        format!("{} {}", name, coll.join(" "))
                    }
                },
            }
        }
    }

    struct Namer {
        value: char,
        set: HashMap<usize, String>,
    }

    impl Namer {
        fn next(&mut self) -> String {
            let v = self.value;
            self.value = ((self.value as u8) + 1) as char;
            format!("{}", v)
        }

        fn name(&mut self, t: usize) -> String {
            let k = { self.set.get(&t).map(|x| x.clone()) };
            if let Some(val) = k {
                val.clone()
            } else {
                let v = self.next();
                self.set.insert(t, v.clone());
                v
            }
        }
    }

    pub fn new_lambda<'a>(v: &'static str, body: Syntax<'a>) -> Syntax<'a> {
        Syntax::Lambda(v, Box::new(body))
    }

    pub fn new_apply<'a>(func: Syntax<'a>, arg: Syntax<'a>) -> Syntax<'a> {
        Syntax::Apply(Box::new(func), Box::new(arg))
    }

    pub fn new_let<'a>(v: &'static str, defn: Syntax<'a>, body: Syntax<'a>) -> Syntax<'a> {
        Syntax::Let(v, Box::new(defn), Box::new(body))
    }

    pub fn new_letrec<'a>(v: &'static str, defn: Syntax<'a>, body: Syntax<'a>) -> Syntax<'a> {
        Syntax::Letrec(v, Box::new(defn), Box::new(body))
    }

    fn test_env<'a>() -> (Vec<Type<'a>>, HashMap<&'a str, usize>) {
        let mut a = vec![
            Type::Operator("int", vec![]),
            Type::Operator("bool", vec![]),
        ];
        let var1 = new_variable(&mut a);
        let var2 = new_variable(&mut a);
        let pair_type = new_operator(&mut a, "*", &[var1, var2]);

        let var3 = new_variable(&mut a);

        let mut env = HashMap::new();

        let right = new_operator(&mut a, "->", &[var2, pair_type]);
        env.insert("pair", new_operator(&mut a, "->", &[var1, right]));
        env.insert("true", 1);
        let right = new_operator(&mut a, "->", &[var3, var3]);
        let right = new_operator(&mut a, "->", &[var3, right]);
        env.insert("cond", new_operator(&mut a, "->", &[1, right]));
        env.insert("zero", new_operator(&mut a, "->", &[0, 1]));
        env.insert("pred", new_operator(&mut a, "->", &[0, 0]));
        let right = new_operator(&mut a, "->", &[0, 0]);
        env.insert("times", new_operator(&mut a, "->", &[0, right]));

        (a, env)
    }

    #[test]
    fn test_factorial() {
        let (mut a, my_env) = test_env();

        // factorial
        let syntax = new_letrec(
            "factorial", // letrec factorial =
            new_lambda(
                "n", // fn n =>
                new_apply(
                    new_apply(
                        // cond (zero n) 1
                        new_apply(
                            Syntax::Identifier("cond"), // cond (zero n)
                            new_apply(Syntax::Identifier("zero"), Syntax::Identifier("n")),
                        ),
                        Syntax::Identifier("1"),
                    ),
                    new_apply(
                        // times n
                        new_apply(Syntax::Identifier("times"), Syntax::Identifier("n")),
                        new_apply(
                            Syntax::Identifier("factorial"),
                            new_apply(Syntax::Identifier("pred"), Syntax::Identifier("n")),
                        ),
                    ),
                ),
            ), // in
            new_apply(Syntax::Identifier("factorial"), Syntax::Identifier("5")),
        );

        let t = analyse(&mut a, &syntax, &my_env, &HashSet::new());
        assert_eq!(
            a[t].as_string(
                &a,
                &mut Namer {
                    value: 'a',
                    set: HashMap::new(),
                }
            ),
            r#"int"#
        );
    }

    #[should_panic]
    #[test]
    fn test_mismatch() {
        let (mut a, my_env) = test_env();

        // fn x => (pair(x(3) (x(true)))
        let syntax = new_lambda(
            "x",
            new_apply(
                new_apply(
                    Syntax::Identifier("pair"),
                    new_apply(Syntax::Identifier("x"), Syntax::Identifier("3")),
                ),
                new_apply(Syntax::Identifier("x"), Syntax::Identifier("true")),
            ),
        );

        let _ = analyse(&mut a, &syntax, &my_env, &HashSet::new());
    }

    #[should_panic]
    #[test]
    fn test_pair() {
        let (mut a, my_env) = test_env();

        // pair(f(3), f(true))
        let syntax = new_apply(
            new_apply(
                Syntax::Identifier("pair"),
                new_apply(Syntax::Identifier("f"), Syntax::Identifier("4")),
            ),
            new_apply(Syntax::Identifier("f"), Syntax::Identifier("true")),
        );

        let _ = analyse(&mut a, &syntax, &my_env, &HashSet::new());
    }

    #[test]
    fn test_mul() {
        let (mut a, my_env) = test_env();

        let pair = new_apply(
            new_apply(
                Syntax::Identifier("pair"),
                new_apply(Syntax::Identifier("f"), Syntax::Identifier("4")),
            ),
            new_apply(Syntax::Identifier("f"), Syntax::Identifier("true")),
        );

        // let f = (fn x => x) in ((pair (f 4)) (f true))
        let syntax = new_let("f", new_lambda("x", Syntax::Identifier("x")), pair);

        let t = analyse(&mut a, &syntax, &my_env, &HashSet::new());
        assert_eq!(
            a[t].as_string(
                &a,
                &mut Namer {
                    value: 'a',
                    set: HashMap::new(),
                }
            ),
            r#"(int * bool)"#
        );
    }

    #[should_panic]
    #[test]
    fn test_recursive() {
        let (mut a, my_env) = test_env();

        // fn f => f f (fail)
        let syntax = new_lambda(
            "f",
            new_apply(Syntax::Identifier("f"), Syntax::Identifier("f")),
        );

        let t = analyse(&mut a, &syntax, &my_env, &HashSet::new());
        assert_eq!(
            a[t].as_string(
                &a,
                &mut Namer {
                    value: 'a',
                    set: HashMap::new(),
                }
            ),
            r#"int"#
        );
    }

    #[test]
    fn test_int() {
        let (mut a, my_env) = test_env();

        // let g = fn f => 5 in g g
        let syntax = new_let(
            "g",
            new_lambda("f", Syntax::Identifier("5")),
            new_apply(Syntax::Identifier("g"), Syntax::Identifier("g")),
        );

        let t = analyse(&mut a, &syntax, &my_env, &HashSet::new());
        assert_eq!(
            a[t].as_string(
                &a,
                &mut Namer {
                    value: 'a',
                    set: HashMap::new(),
                }
            ),
            r#"int"#
        );
    }

    #[test]
    fn test_generic_nongeneric() {
        let (mut a, my_env) = test_env();

        // example that demonstrates generic and non-generic variables:
        // fn g => let f = fn x => g in pair (f 3, f true)
        let syntax = new_lambda(
            "g",
            new_let(
                "f",
                new_lambda("x", Syntax::Identifier("g")),
                new_apply(
                    new_apply(
                        Syntax::Identifier("pair"),
                        new_apply(Syntax::Identifier("f"), Syntax::Identifier("3")),
                    ),
                    new_apply(Syntax::Identifier("f"), Syntax::Identifier("true")),
                ),
            ),
        );

        let t = analyse(&mut a, &syntax, &my_env, &HashSet::new());
        assert_eq!(
            a[t].as_string(
                &a,
                &mut Namer {
                    value: 'a',
                    set: HashMap::new(),
                }
            ),
            r#"(a -> (a * a))"#
        );
    }

    #[test]
    fn test_composition() {
        let (mut a, my_env) = test_env();

        // Function composition
        // fn f (fn g (fn arg (f g arg)))
        let syntax = new_lambda(
            "f",
            new_lambda(
                "g",
                new_lambda(
                    "arg",
                    new_apply(
                        Syntax::Identifier("g"),
                        new_apply(Syntax::Identifier("f"), Syntax::Identifier("arg")),
                    ),
                ),
            ),
        );

        let t = analyse(&mut a, &syntax, &my_env, &HashSet::new());
        assert_eq!(
            a[t].as_string(
                &a,
                &mut Namer {
                    value: 'a',
                    set: HashMap::new(),
                }
            ),
            r#"((a -> b) -> ((b -> c) -> (a -> c)))"#
        );
    }

    #[test]
    fn test_fun() {
        let (mut a, my_env) = test_env();

        // Function composition
        // fn f (fn g (fn arg (f g arg)))
        let syntax = new_lambda(
            "f",
            new_lambda(
                "g",
                new_lambda(
                    "arg",
                    new_apply(
                        Syntax::Identifier("g"),
                        new_apply(Syntax::Identifier("f"), Syntax::Identifier("arg")),
                    ),
                ),
            ),
        );

        let t = analyse(&mut a, &syntax, &my_env, &HashSet::new());
        assert_eq!(
            a[t].as_string(
                &a,
                &mut Namer {
                    value: 'a',
                    set: HashMap::new(),
                }
            ),
            r#"((a -> b) -> ((b -> c) -> (a -> c)))"#
        );
    }
}
