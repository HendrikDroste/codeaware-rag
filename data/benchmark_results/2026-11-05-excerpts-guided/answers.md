# Benchmark answers

Source links normalized to the pinned Flask revision; raw answers are unchanged.

## R1 Q11 baseline

The blueprint class hierarchy is a single inheritance chain:

```text
object
└── flask.sansio.scaffold.Scaffold
    └── flask.sansio.blueprints.Blueprint
        └── flask.blueprints.Blueprint  (public flask.Blueprint)
```

- **`Scaffold`** implicitly inherits from Python’s `object`. It provides behavior shared by applications and blueprints, including static/template paths and resource configuration. See [scaffold.py:52](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L52).
- **Sans-I/O `Blueprint`** explicitly inherits from `Scaffold`. It collects routes and application functions for deferred registration. See [sansio/blueprints.py:119](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L119).
- **Public `Blueprint`** inherits from that Sans-I/O class, imported under the alias `SansioBlueprint`. It adds CLI initialization, static-file serving, and resource-file opening. See [blueprints.py:10](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/blueprints.py#L10), [blueprints.py:18](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/blueprints.py#L18). This is exported as `flask.Blueprint` in [__init__.py:3](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/__init__.py#L3).

`BlueprintSetupState` is a separate helper class, outside this inheritance chain ([sansio/blueprints.py:34](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L34)).

There is no uncertainty about these declared inheritance relationships; this describes the repository’s classes, excluding subclasses external users may define.

## R1 Q11 codeaware

The blueprint inheritance chain is:

```text
object
└── flask.sansio.scaffold.Scaffold
    └── flask.sansio.blueprints.Blueprint
        └── flask.blueprints.Blueprint  (public flask.Blueprint)
            └── MyBlueprint            (test subclass)
```

- **`Scaffold`** has no explicit base, so inherits from Python’s `object`. It provides common behavior shared by applications and blueprints: [scaffold.py:52](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L52).
- **Sans-I/O `Blueprint`** directly inherits `Scaffold`. It represents routes and application functions recorded for later registration: [sansio/blueprints.py:119](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L119).
- **Public `Blueprint`** directly inherits that class, imported under the alias `SansioBlueprint`: [blueprints.py:10](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/blueprints.py#L10), [blueprints.py:18](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/blueprints.py#L18). It is exported as `flask.Blueprint`: [__init__.py:3](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/__init__.py#L3).
- The indexed test subclass **`MyBlueprint`** overrides `get_send_file_max_age` to return `100`: [test_blueprints.py:222](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/tests/test_blueprints.py#L222).

The production chain is confirmed by source declarations. The inheritance index is partial, so it cannot establish that no additional subclasses exist.

## R1 Q16 codeaware

`route` is a convenience decorator around `add_url_rule`. These registrations are equivalent:

```python
@app.route("/hello", methods=["GET"])
def hello():
    return "Hello"
```

```python
def hello():
    return "Hello"

app.add_url_rule("/hello", view_func=hello, methods=["GET"])
```

The decorator extracts the optional endpoint, calls `self.add_url_rule(rule, endpoint, f, **options)`, and returns the original function. It registers the function without wrapping its execution. See [scaffold.py:368](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L368).

Under the hood, the app’s `add_url_rule` does four main things:

1. **Chooses an endpoint name.** If none is supplied, it uses `view_func.__name__`. The endpoint is the key connecting a URL rule to its handler. See [app.py:614](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L614) and [scaffold.py:709](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L709).
2. **Determines allowed HTTP methods.** It uses explicit `methods`, otherwise the function’s `methods` attribute, otherwise `GET`. It uppercases them, merges `required_methods`, and decides whether to add automatic `OPTIONS` using the argument, function attribute, and app configuration. See [app.py:617](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L617).
3. **Registers the routing rule.** It constructs `self.url_rule_class(...)`, attaches the automatic-options flag, and adds the rule to `self.url_map`. The default rule class is Werkzeug’s `Rule`. See [app.py:651](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L651) and [app.py:255](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L255).
4. **Registers the handler separately.** It stores `view_functions[endpoint] = view_func`, rejecting an endpoint already associated with a different function. Multiple rules can therefore share the same endpoint and function. See [app.py:655](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L655).

When a request arrives, dispatch uses the matched rule’s endpoint to retrieve that function and calls it with the matched URL arguments. Automatic `OPTIONS` requests take a separate response path. See [app.py:1011](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1011).

You can also register a rule with an explicit endpoint and supply its function later through `@app.endpoint(...)`; this reflects the separation between routing rules and handlers. See [scaffold.py:412](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L412).

**Evidence boundary:** Flask’s docstring documents automatic `HEAD` support, but the implementation delegates rule construction to Werkzeug; I did not inspect that dependency’s internals. See [scaffold.py:408](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L408).

## R1 Q16 baseline

`route` is a convenience decorator around `add_url_rule`, which performs the actual registration. The decorator extracts `endpoint`, calls `self.add_url_rule(rule, endpoint, f, **options)`, and returns the original function unchanged ([scaffold.py:368](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L368)).

These therefore register the same route:

```python
@app.route("/hello", methods=["GET"])
def hello():
    return "Hello"
```

```python
def hello():
    return "Hello"

app.add_url_rule("/hello", view_func=hello, methods=["GET"])
```

Under the hood, the application’s `add_url_rule`:

1. **Chooses an endpoint name.** If omitted, it uses `view_func.__name__`. The endpoint identifies the handler independently of the URL ([app.py:614](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L614), [scaffold.py:709](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L709)).
2. **Determines HTTP methods.** It uses explicit `methods`, otherwise `view_func.methods`, otherwise `GET`. It rejects a bare string, uppercases method names, and includes `view_func.required_methods` ([app.py:617](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L617)).
3. **Configures automatic `OPTIONS`.** An explicit argument takes precedence over the function’s `provide_automatic_options` attribute; otherwise Flask enables it when `OPTIONS` is absent and `PROVIDE_AUTOMATIC_OPTIONS` is enabled ([app.py:634](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L634)).
4. **Registers the routing rule and handler separately.** It constructs `self.url_rule_class`—Werkzeug’s `Rule` by default—and adds it to `self.url_map`. It then stores the function in `self.view_functions[endpoint]`. Reusing an endpoint with a different function raises `AssertionError`; registering the same function again is allowed ([app.py:251](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L251), [app.py:651](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L651)).

At request time, dispatch uses the matched rule’s endpoint to retrieve the function, then calls it through `ensure_sync` with the captured URL arguments. Automatic `OPTIONS` requests are handled before that call ([app.py:1007](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1007)).

Both APIs are setup operations: Flask rejects them after the application has handled its first request ([scaffold.py:42](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L42), [app.py:411](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L411)).

One evidence boundary: Flask documents automatic `HEAD` support ([scaffold.py:408](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L408)), but this implementation delegates rule construction to Werkzeug. I did not inspect that dependency’s implementation.

## R1 Q18 baseline

Flask creates the request object from the **WSGI environment** supplied by the server, before dispatching to your view:

1. The server calls `Flask.__call__(environ, start_response)`, which delegates to `wsgi_app()`. That calls `self.request_context(environ)`. [app.py:1653](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1653), [app.py:1679](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1679)
2. In this checkout, `request_context()` delegates to `AppContext.from_environ()`. Its key construction step is:
   ```python
   request = app.request_class(environ)
   request.json_module = app.json
   return cls(app, request=request)
   ```
   Thus each invocation creates a request and stores it in a new context. [app.py:1576](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1576), [ctx.py:340](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/ctx.py#L340)
3. `app.request_class` defaults to Flask’s `Request`, a subclass of Werkzeug’s `Request`. Applications can replace it with a custom subclass. [app.py:237](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L237), [wrappers.py:18](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/wrappers.py#L18)
4. `wsgi_app()` pushes the context before dispatching. Pushing binds it to a `ContextVar`; `flask.request` is a `LocalProxy` resolving to that context’s request object. [app.py:1657](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1657), [ctx.py:433](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/ctx.py#L433), [globals.py:57](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/globals.py#L57)

There is no uncertainty about this construction path in the checked-out code. Werkzeug’s internal initialization was not inspected; the explanation covers Flask’s side.

## R1 Q18 codeaware

Flask creates the request from the **WSGI environment supplied by the server**, before dispatching to your view:

1. The server calls `Flask.__call__(environ, start_response)`, which forwards to `wsgi_app()` ([app.py:1679](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1679)).
2. `wsgi_app()` calls `self.request_context(environ)`. That method delegates to `AppContext.from_environ(self, environ)` ([app.py:1653](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1653), [app.py:1576](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1576)).
3. **The actual construction is `request = app.request_class(environ)`.** Flask then assigns the app’s JSON provider and stores the request in a new context ([ctx.py:340](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/ctx.py#L340)).
4. `wsgi_app()` pushes that context before dispatching the request. The public `flask.request` is a `LocalProxy` that resolves to the active context’s request ([app.py:1657](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1657), [globals.py:57](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/globals.py#L57)).

By default, `request_class` is Flask’s `Request`, a subclass of Werkzeug’s request wrapper. Applications can replace it with a custom subclass ([app.py:239](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L239), [wrappers.py:18](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/wrappers.py#L18)).

This describes the checked-out repository, which uses a combined `AppContext`; older Flask versions may organize the context differently. Werkzeug’s internal environment parsing was not inspected.

## R2 Q11 codeaware

The blueprint inheritance chain is:

```text
object
└── flask.sansio.scaffold.Scaffold
    └── flask.sansio.blueprints.Blueprint
        └── flask.blueprints.Blueprint  (public flask.Blueprint)
            └── MyBlueprint  (test subclass)
```

- **`Scaffold`** has no explicit superclass, so implicitly inherits from `object`. It provides common behavior for applications and blueprints ([src/flask/sansio/scaffold.py:52](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L52)).
- **Sans-I/O `Blueprint(Scaffold)`** implements the blueprint abstraction: recording routes and other application functions for later registration ([src/flask/sansio/blueprints.py:119](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L119)).
- **Public `Blueprint(SansioBlueprint)`** extends that class. `SansioBlueprint` is an import alias, and this concrete class is exported as `flask.Blueprint` ([src/flask/blueprints.py:10](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/blueprints.py#L10), [definition:18](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/blueprints.py#L18), [export:3](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/__init__.py#L3)).
- **Test subclass `MyBlueprint`** inherits from `flask.Blueprint` and overrides `get_send_file_max_age` to return `100` ([tests/test_blueprints.py:222](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/tests/test_blueprints.py#L222)).

The production chain is explicit in the source. The static graph identified the test subclass shown above, but its partial indexing cannot establish that no other subclasses exist.

## R2 Q11 baseline

Assuming you mean Python class inheritance, the blueprint hierarchy is:

```text
object
└── flask.sansio.scaffold.Scaffold
    └── flask.sansio.blueprints.Blueprint
        └── flask.blueprints.Blueprint  (public flask.Blueprint)
```

- **`Scaffold`** provides common behavior shared by applications and blueprints. It implicitly inherits from `object`: [src/flask/sansio/scaffold.py:52](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L52).
- **Sans-I/O `Blueprint`** inherits `Scaffold` and implements deferred registration of routes and other application functions: [src/flask/sansio/blueprints.py:119](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L119).
- **Public `Blueprint`** inherits the Sans-I/O class, imported as `SansioBlueprint`, and adds concrete functionality such as CLI setup and static-file serving: [src/flask/blueprints.py:10](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/blueprints.py#L10), [src/flask/blueprints.py:18](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/blueprints.py#L18). It is re-exported as `flask.Blueprint`: [src/flask/__init__.py:3](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/__init__.py#L3).

If you meant **nested blueprints**, that is a registration relationship: `register_blueprint()` stores a child blueprint in the parent’s `_blueprints` list ([src/flask/sansio/blueprints.py:256](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L256)). The class inheritance above is explicit; the only uncertainty is which meaning of “tree” you intended.

## R2 Q16 baseline

`add_url_rule` registers a URL pattern and associates its endpoint with a view function. `route` is a convenience decorator that calls it.

These are equivalent:

```python
@app.route("/hello/<name>", methods=["GET"])
def hello(name):
    return f"Hello, {name}"
```

```python
def hello(name):
    return f"Hello, {name}"

app.add_url_rule("/hello/<name>", view_func=hello, methods=["GET"])
```

The decorator extracts the optional `endpoint`, calls `self.add_url_rule(rule, endpoint, f, **options)`, and returns the original function—it doesn’t wrap the view’s execution. See [scaffold.py:368](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L368).

Under the hood, `add_url_rule`:

1. **Chooses an endpoint name.** An explicit `endpoint` wins; otherwise it uses `view_func.__name__`. The endpoint is the key connecting routing to the callable. See [app.py:614](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L614) and [scaffold.py:709](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L709).
2. **Resolves HTTP methods.** It uses the `methods` argument, then the view’s `methods` attribute, then defaults to `GET`. It rejects a single string, uppercases the method names, and merges the view’s `required_methods`. Automatic `OPTIONS` handling depends on the explicit argument, the view’s `provide_automatic_options` attribute, and application configuration. See [app.py:617](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L617).
3. **Creates and stores the routing rule.** It constructs `self.url_rule_class(...)` with the URL, methods, endpoint, and remaining options, attaches the automatic-OPTIONS flag, and adds it to `self.url_map`. See [app.py:651](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L651).
4. **Stores the callable separately.** It sets `self.view_functions[endpoint] = view_func`. Reusing an endpoint for a different function raises `AssertionError`; registering the same function again is allowed. You can also omit the callable and associate it later. See [app.py:655](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L655) and [scaffold.py:412](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L412).

When a request reaches dispatch, Flask uses the matched rule’s endpoint to look up the view and calls it with the matched URL arguments through `ensure_sync`. Automatic `OPTIONS` requests receive a default response instead. See [app.py:1011](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1011).

Both registration APIs enforce setup timing: calling them after the application has handled its first request raises an error. See [scaffold.py:42](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L42) and [app.py:411](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L411).

Evidence limit: this repository documents automatic `HEAD` support ([scaffold.py:408](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L408)), but delegates rule construction to Werkzeug; I did not inspect that dependency’s implementation.

## R2 Q16 codeaware

`route` is a convenience decorator around `add_url_rule`. It registers the decorated function, then returns that same function unchanged ([scaffold.py:368](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L368)):

```python
@app.route("/hello", methods=["GET"])
def hello():
    return "Hello"
```

is equivalent to:

```python
def hello():
    return "Hello"

app.add_url_rule("/hello", view_func=hello, methods=["GET"])
```

Under the hood, `add_url_rule` performs these steps:

1. **Choose an endpoint name.** It uses the supplied `endpoint`, or defaults to `view_func.__name__`. The endpoint is the key connecting the routing rule to the function ([app.py:614](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L614), [scaffold.py:709](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L709)).
2. **Determine HTTP methods.** It takes explicit `methods`, otherwise the function’s `methods` attribute, otherwise `GET`. It rejects a bare string, uppercases method names, and adds the function’s `required_methods` ([app.py:617](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L617)).
3. **Configure automatic `OPTIONS`.** An explicit argument takes precedence over the function attribute, then the `PROVIDE_AUTOMATIC_OPTIONS` configuration. When enabled, `OPTIONS` is added to the methods ([app.py:634](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L634)).
4. **Register the routing rule and handler.** It creates a rule using `url_rule_class`—normally Werkzeug’s `Rule`—passes through the remaining options, adds it to `self.url_map`, and stores the function in `self.view_functions[endpoint]`. Reusing an endpoint for a different function raises `AssertionError`; reusing it for the same function is allowed ([app.py:251](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L251), [app.py:651](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L651)).

When a request has matched a rule, dispatch uses its endpoint to retrieve and invoke the registered function with the URL arguments. Automatic `OPTIONS` requests are handled before invoking the function ([app.py:1007](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1007)).

For blueprints, the same decorator calls the blueprint’s `add_url_rule`, which records registration for later; the setup state applies URL and endpoint prefixes before calling the application’s method ([blueprints.py:433](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L433), [blueprints.py:98](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L98)).

**Scope/uncertainty:** These details are supported by the local Flask source. Its documentation says `HEAD` is added automatically; the method delegates rule creation to Werkzeug, whose implementation I did not inspect ([scaffold.py:408](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L408)).

## R2 Q18 codeaware

Flask creates the request object from the incoming **WSGI environment**, before dispatching to a view:

1. The WSGI server calls `Flask.__call__(environ, start_response)`, which forwards to `wsgi_app`. [src/flask/app.py:1679](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1679)
2. `wsgi_app` calls `self.request_context(environ)`, which delegates to `AppContext.from_environ(self, environ)`. [src/flask/app.py:1653](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1653), [src/flask/app.py:1576](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1576)
3. The actual construction happens there:
   ```python
   request = app.request_class(environ)
   request.json_module = app.json
   return cls(app, request=request)
   ```
   [src/flask/ctx.py:346](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/ctx.py#L346)
4. Flask pushes that context before dispatching. Application code accesses its request through the `flask.request` context-local proxy. [src/flask/app.py:1657](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1657), [src/flask/globals.py:57](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/globals.py#L57)

By default, `request_class` is Flask’s `Request`, a subclass of Werkzeug’s request wrapper. Applications can replace it with a custom subclass. [src/flask/app.py:239](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L239), [src/flask/wrappers.py:18](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/wrappers.py#L18)

This creation path is explicit in the local source. Werkzeug’s internal initialization was not inspected.

## R2 Q18 baseline

Flask constructs a fresh request object from the **WSGI environment** for each incoming request:

1. The WSGI server calls `Flask.__call__(environ, start_response)`, which forwards to `wsgi_app`. [app.py:1679](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1679)
2. `wsgi_app` calls `self.request_context(environ)`, which delegates to `AppContext.from_environ(self, environ)`. [app.py:1653](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1653), [app.py:1576](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1576)
3. The actual construction happens here:
   ```python
   request = app.request_class(environ)
   request.json_module = app.json
   return cls(app, request=request)
   ```
   [ctx.py:346](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/ctx.py#L346)

By default, `request_class` is Flask’s `Request`, a subclass of Werkzeug’s request wrapper. Applications can replace it with a custom subclass. [app.py:239](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L239), [wrappers.py:18](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/wrappers.py#L18)

Flask then pushes the context before dispatching the request. The public `flask.request` is a `LocalProxy` that resolves to the request stored in the active context. [app.py:1657](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1657), [globals.py:57](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/globals.py#L57)

This flow is directly supported by this checkout. Werkzeug’s internal processing of `environ` was not inspected.

## R3 Q11 baseline

The blueprint class inheritance chain is:

```text
object
└── flask.sansio.scaffold.Scaffold
    └── flask.sansio.blueprints.Blueprint
        └── flask.blueprints.Blueprint  (public flask.Blueprint)
```

- **`Scaffold`** provides behavior shared by applications and blueprints. It has no explicit base, so inherits from Python’s `object`: [src/flask/sansio/scaffold.py:52](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L52).
- **Sans-I/O `Blueprint(Scaffold)`** adds blueprint configuration and deferred registration of application functions: [src/flask/sansio/blueprints.py:119](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L119).
- **Public `Blueprint(SansioBlueprint)`** extends that class with CLI initialization and concrete file-serving behavior. The base alias is imported at line 10, and inheritance is declared at line 18: [src/flask/blueprints.py:10](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/blueprints.py#L10), [src/flask/blueprints.py:18](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/blueprints.py#L18).
- **`flask.Blueprint`** re-exports this public class: [src/flask/__init__.py:3](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/__init__.py#L3).

The tests demonstrate further subclassing with `MyBlueprint(flask.Blueprint)`, overriding `get_send_file_max_age`: [tests/test_blueprints.py:222](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/tests/test_blueprints.py#L222).

There is no uncertainty about this declared inheritance chain. This describes class inheritance; application-specific blueprint nesting is a separate relationship.

## R3 Q11 codeaware

The blueprint inheritance chain is:

```text
object
└── flask.sansio.scaffold.Scaffold
    └── flask.sansio.blueprints.Blueprint
        └── flask.blueprints.Blueprint
            └── MyBlueprint (test subclass)
```

- **`Scaffold`** has no explicit base, so it inherits from Python’s `object`. It provides behavior shared by Flask applications and blueprints ([scaffold.py:52](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L52)).
- **Sans-I/O `Blueprint`** inherits from `Scaffold`. It records routes and other application functions for later registration ([sansio/blueprints.py:119](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L119)).
- **Concrete `Blueprint`** inherits from that Sans-I/O class, imported under the alias `SansioBlueprint` ([blueprints.py:10](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/blueprints.py#L10), [blueprints.py:18](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/blueprints.py#L18)).
- **`MyBlueprint`**, defined inside a test, extends `flask.Blueprint` and overrides `get_send_file_max_age` ([test_blueprints.py:222](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/tests/test_blueprints.py#L222)).

The declared chain is explicit in the source. The static index identified the test subclass shown above, but may omit dynamically created or otherwise unresolved subclasses.

## R3 Q16 codeaware

`route` is a convenience decorator around `add_url_rule`. The latter does the actual registration: it connects a URL pattern to an **endpoint name**, then connects that endpoint to a **view function**.

These are equivalent:

```python
@app.route("/hello/<name>", methods=["GET"])
def hello(name):
    return f"Hello, {name}!"
```

```python
def hello(name):
    return f"Hello, {name}!"

app.add_url_rule("/hello/<name>", view_func=hello, methods=["GET"])
```

The decorator takes the optional `endpoint` out of its options, calls `self.add_url_rule(rule, endpoint, f, **options)`, and returns the original function unchanged. It registers the function rather than wrapping its execution. See [src/flask/sansio/scaffold.py:368](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L368).

Under the hood, `add_url_rule`:

1. **Chooses the endpoint.** If none is supplied, it uses `view_func.__name__`, so the example’s endpoint is `"hello"`. See [src/flask/sansio/app.py:614](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L614) and [src/flask/sansio/scaffold.py:709](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L709).

2. **Determines allowed HTTP methods.** It uses the explicit `methods` option, otherwise `view_func.methods`, otherwise `GET`. It rejects a single string, uppercases the method names, and merges any `view_func.required_methods`. Automatic `OPTIONS` handling is controlled by the explicit argument, the function attribute, or the `PROVIDE_AUTOMATIC_OPTIONS` configuration. See [src/flask/sansio/app.py:617](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L617).

3. **Creates and stores the routing rule.** It constructs `self.url_rule_class(rule, methods=methods, **options)`, attaches the automatic-options flag, and adds it to `self.url_map`. The default rule class is Werkzeug’s `Rule`. See [src/flask/sansio/app.py:255](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L255) and [src/flask/sansio/app.py:651](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L651).

4. **Stores the callable separately.** It assigns `self.view_functions[endpoint] = view_func`. Registering a different function under an existing endpoint raises `AssertionError`; reusing the same function is allowed. You can also register a rule with an explicit endpoint and attach its function later. See [src/flask/sansio/app.py:655](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L655) and [src/flask/sansio/scaffold.py:411](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L411).

When a request arrives, Flask takes the matched rule’s endpoint, looks up the function in `view_functions`, and calls it with the extracted URL arguments. Automatic `OPTIONS` requests are handled before invoking the view. See [src/flask/app.py:1011](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1011).

One scope limitation: Flask’s documentation says `HEAD` is added automatically, but the shown method delegates rule construction to Werkzeug; I have not inspected that dependency’s implementation. See [src/flask/sansio/scaffold.py:407](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L407).

## R3 Q16 baseline

`route()` is a convenience decorator around `add_url_rule()`. The latter performs the actual registration; it does not execute the view.

These are equivalent:

```python
@app.route("/hello", methods=["GET"])
def hello():
    return "Hello!"
```

```python
def hello():
    return "Hello!"

app.add_url_rule("/hello", view_func=hello, methods=["GET"])
```

The decorator extracts the optional `endpoint`, calls `self.add_url_rule(rule, endpoint, f, **options)`, and returns the original function unchanged. [src/flask/sansio/scaffold.py:368](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L368)

Under the hood, the application’s `add_url_rule()`:

1. **Chooses an endpoint name.** An explicit `endpoint` wins; otherwise it uses `view_func.__name__`. The endpoint is the key linking a URL rule to its handler. [src/flask/sansio/app.py:614](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L614), [src/flask/sansio/scaffold.py:709](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L709)
2. **Determines HTTP methods.** It uses the `methods` option, then the view’s `methods` attribute, then `GET`. It rejects a single string, uppercases method names, and merges the view’s `required_methods`. Automatic `OPTIONS` handling depends on the explicit argument, the view’s `provide_automatic_options` attribute, and finally application configuration. [src/flask/sansio/app.py:617](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L617)
3. **Registers the routing rule.** It constructs `self.url_rule_class(...)`, attaches the automatic-options flag, and adds the rule to `self.url_map`. Remaining options are forwarded to the rule constructor. [src/flask/sansio/app.py:651](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L651)
4. **Registers the callable separately.** It stores `view_functions[endpoint] = view_func`. Reusing an endpoint for a different function raises `AssertionError`; multiple rules can share the same function and endpoint. [src/flask/sansio/app.py:655](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L655)

At request time, Flask uses the matched rule’s endpoint to retrieve and call the view, passing URL variables as keyword arguments. Automatic `OPTIONS` requests can be answered before calling the view. [src/flask/app.py:1011](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1011)

Both APIs are setup operations: registration is rejected after the application has handled its first request. [src/flask/sansio/scaffold.py:42](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L42), [src/flask/sansio/app.py:411](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L411)

One boundary: Flask documents automatic `HEAD` support, but delegates rule construction to Werkzeug. I have not inspected Werkzeug’s implementation, so that detail is supported here by Flask’s documentation rather than dependency source. [src/flask/sansio/scaffold.py:408](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L408)

## R3 Q18 baseline

Flask creates the request object from the **WSGI environment** supplied by the server:

1. The server calls `Flask.__call__(environ, start_response)`, which delegates to `wsgi_app`. That method calls `self.request_context(environ)` before dispatching the request. ([app.py:1679](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1679), [app.py:1653](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1653))
2. `request_context` calls `AppContext.from_environ`. The actual creation happens there:
   ```python
   request = app.request_class(environ)
   request.json_module = app.json
   return cls(app, request=request)
   ```
   Thus each normal incoming request gets a new request object stored in its context. ([app.py:1576](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1576), [ctx.py:340](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/ctx.py#L340))
3. By default, `app.request_class` is Flask’s `Request`, a subclass of Werkzeug’s `Request`. Applications can replace it with a custom subclass. ([app.py:239](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L239), [wrappers.py:18](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/wrappers.py#L18))
4. Flask pushes the context before dispatch. This makes the object accessible through `flask.request`, a `LocalProxy` resolving the active context’s `request` attribute. ([app.py:1657](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1657), [ctx.py:433](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/ctx.py#L433), [globals.py:57](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/globals.py#L57))

These steps are directly supported by this checkout. Werkzeug’s internal initialization and parsing behavior were not inspected.

## R3 Q18 codeaware

Flask creates a fresh request object from the WSGI environment for each incoming request:

1. The WSGI server calls `Flask.__call__(environ, start_response)`, which forwards to `wsgi_app`. [src/flask/app.py:1679](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1679)
2. `wsgi_app` calls `self.request_context(environ)`, which delegates to `AppContext.from_environ(self, environ)`. [src/flask/app.py:1653](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1653), [src/flask/app.py:1576](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1576)
3. The actual construction happens here:
   ```python
   request = app.request_class(environ)
   request.json_module = app.json
   return cls(app, request=request)
   ```
   [src/flask/ctx.py:346](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/ctx.py#L346)
4. Flask pushes the context before dispatching the request. The familiar `flask.request` is a `LocalProxy` that accesses that context’s request object. [src/flask/app.py:1657](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1657), [src/flask/globals.py:57](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/globals.py#L57)

By default, `app.request_class` is Flask’s `Request`, a subclass of Werkzeug’s `Request`; applications can replace it with a custom subclass. [src/flask/app.py:239](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L239), [src/flask/wrappers.py:18](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/wrappers.py#L18)

This flow is explicit in the local source. The application-specific uncertainty is whether an app overrides `request_class` or the entry points. This checkout uses a combined `AppContext` for request handling.
