# %%
import javalang

# %%
def parser_helper(code, format):
    formats = [
        'java_file',
        'java_class',
        'java_method',
    ]
    assert format in formats, 'format must be one of {}'.format(formats)

    try:
        if format == 'java_file' or format == 'java_class':
            toks = javalang.tokenizer.tokenize(code)
            tree = javalang.parser.Parser(toks).parse()
            return [node for path, node in tree.filter(javalang.tree.MethodDeclaration)]
        if format == 'java_method':
            code = "class A { " + code + " }"
            toks = javalang.tokenizer.tokenize(code)
            tree = javalang.parser.Parser(toks).parse()
            return [node for path, node in tree.filter(javalang.tree.MethodDeclaration)]
    except Exception as e:
        raise e
        return None

# %%

assert len(parser_helper("public void foo() {}", 'java_method')) == 1

assert len(parser_helper("class A { public void foo() {} }", 'java_class')) == 1

assert len(parser_helper("class A { public void foo() {} public void bar() {} }", 'java_class')) == 2

# %%
class MethodFeaturizer:
    @classmethod
    def from_code(cls, code, force_list=False):
        methods = parser_helper(code, 'java_method')
        if len(methods) > 1:
            return [cls(m) for m in methods]
        elif len(methods) == 1:
            return cls(methods[0])
    
    def __init__(self, ast: javalang.tree.MethodDeclaration):
        # self.toks = ast.tokens
        self.ast = ast
    
    def has_if_statement(self):
        return list(self.ast.filter(javalang.tree.IfStatement)) != []
    
    def has_while_statement(self):
        return list(self.ast.filter(javalang.tree.WhileStatement)) != []
    
    def has_for_statement(self):
        return list(self.ast.filter(javalang.tree.ForStatement)) != []
    
    def has_switch_statement(self):
        return list(self.ast.filter(javalang.tree.SwitchStatement)) != []
    
    def has_try_catch_statement(self):
        return list(self.ast.filter(javalang.tree.TryStatement)) != []
    
    def has_throw_statement(self):
        return list(self.ast.filter(javalang.tree.ThrowStatement)) != [] or \
            list(self.ast.throws) != []
    
    def has_function_call(self):
        return list(self.ast.filter(javalang.tree.MethodInvocation)) != []

# %%
f = MethodFeaturizer.from_code('class FooTest { public void foo() { if (true) { System.out.println("Hello"); } } }')
assert f.has_if_statement()

# %%
f = MethodFeaturizer.from_code('class FooTest { public void foo() { while (true) { System.out.println("Hello"); } } }')
assert f.has_while_statement()

# %%
f = MethodFeaturizer.from_code('class FooTest { public void foo() { for (int i = 0; i < 10; i++) { System.out.println("Hello"); } } }')
assert f.has_for_statement()

# %%
f = MethodFeaturizer.from_code('class FooTest { public void foo() { switch (1) { case 1: System.out.println("Hello"); } } }')
assert f.has_switch_statement()

# %%
f = MethodFeaturizer.from_code('class FooTest { public void foo() { try { System.out.println("Hello"); } catch (Exception e) { } } }')
assert f.has_try_catch_statement()

# %%
f = MethodFeaturizer.from_code('class FooTest { public void foo() { throw new Exception(); } }')
assert f.has_throw_statement()

# %%
f = MethodFeaturizer.from_code('class FooTest { public void foo() { System.out.println("Hello"); } }')
assert not f.has_if_statement()

# %%
f = MethodFeaturizer.from_code('class FooTest { public void foo() { System.out.println("Hello"); } }')
assert f.has_function_call()

# %%



