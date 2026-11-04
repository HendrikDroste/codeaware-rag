# Pilot answers

Source links are normalized to the pinned Flask revision. Raw answers remain in the recorded run directories.

## auto: q11-baseline

The blueprint class inheritance tree is:

```text
object
└── flask.sansio.scaffold.Scaffold
    └── flask.sansio.blueprints.Blueprint
        └── flask.blueprints.Blueprint  (exported as flask.Blueprint)
```

- **`Scaffold`** has no explicit base class, so it inherits from Python’s `object`. It provides common behavior for applications and blueprints, including resource paths and static/template configuration ([scaffold.py:52](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L52)).
- **The Sans-IO `Blueprint`** directly inherits from `Scaffold`. It records deferred setup functions and manages blueprint registration and nesting ([sansio/blueprints.py:119](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L119), [sansio/blueprints.py:204](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L204)).
- **The public `Blueprint`** inherits from that Sans-IO class, imported under the alias `SansioBlueprint`. It adds CLI initialization, static-file serving, and resource-file opening ([blueprints.py:10](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/blueprints.py#L10), [blueprints.py:18](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/blueprints.py#L18)). This is the class exported as `flask.Blueprint` ([__init__.py:3](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/__init__.py#L3)).

`BlueprintSetupState` is a separate helper class, outside this inheritance chain ([sansio/blueprints.py:34](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L34)).

The only ambiguity is whether “inheritance” means class inheritance or nested blueprints. Nesting uses composition: `register_blueprint` stores child blueprint instances; it does not create subclasses ([sansio/blueprints.py:256](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L256)). The class hierarchy above is explicit in the source.

## auto: q11-codeaware

The blueprint class hierarchy is:

```text
object
└── flask.sansio.scaffold.Scaffold
    └── flask.sansio.blueprints.Blueprint
        └── flask.blueprints.Blueprint  (public flask.Blueprint)
```

- **`Scaffold`** supplies shared application/blueprint behavior, including routing decorators and request hooks. It has no explicit superclass, so inherits from `object`. See [scaffold.py:52](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L52).
- **Sans-I/O `Blueprint(Scaffold)`** implements deferred setup and blueprint registration. See [sansio/blueprints.py:119](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L119).
- **Public `Blueprint(SansioBlueprint)`** adds CLI initialization, static-file serving, and resource-file access. Its base is imported under the alias `SansioBlueprint`. See [blueprints.py:10](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/blueprints.py#L10) and [blueprints.py:18](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/blueprints.py#L18). `flask.Blueprint` re-exports this class in [__init__.py:3](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/__init__.py#L3).

`Flask` shares the `Scaffold` ancestor through `App`; it is not a blueprint superclass ([sansio/app.py:60](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L60), [app.py:108](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L108)).

The only ambiguity is whether “inheritance” means nested blueprints: those use registration/composition, with child instances stored in `_blueprints`, rather than class inheritance ([sansio/blueprints.py:256](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L256)). The class hierarchy above is explicit in the local source.

## auto: q16-baseline

`add_url_rule` registers the connection **URL pattern → endpoint name → view function**. The `route` decorator is a convenience wrapper around it.

For example, these registrations are equivalent:

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

`route` returns a decorator that extracts the optional `endpoint`, calls `self.add_url_rule(rule, endpoint, f, **options)`, and returns the original function. It registers the function without wrapping its execution. See [scaffold.py:368](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L368).

Under the hood, the application’s `add_url_rule` does the following:

1. **Chooses the endpoint.** If omitted, it uses `view_func.__name__`, such as `"hello"`. See [app.py:614](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L614) and [scaffold.py:709](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L709).
2. **Resolves HTTP methods.** It uses the supplied `methods`, otherwise the function’s `methods` attribute, otherwise `GET`. It rejects a single string, uppercases method names, and includes the function’s `required_methods`. See [app.py:617](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L617).
3. **Configures automatic `OPTIONS`.** An explicit argument takes precedence over the function’s `provide_automatic_options` attribute. Otherwise, Flask enables it when `OPTIONS` was not explicitly listed and `PROVIDE_AUTOMATIC_OPTIONS` is enabled. See [app.py:634](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L634).
4. **Stores the routing rule and handler separately.** It constructs `self.url_rule_class(...)`—normally Werkzeug’s `Rule`—and adds it to `self.url_map`. If a view was provided, it stores it in `self.view_functions[endpoint]`. Registering a different function for an existing endpoint raises `AssertionError`; multiple rules may share the same function. See [app.py:651](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L651) and [app.py:251](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L251).

When a request arrives, the URL adapter matches a rule and extracts its arguments ([ctx.py:405](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/ctx.py#L405)). Dispatch then looks up `view_functions[rule.endpoint]` and calls it with those arguments as keyword arguments, using `ensure_sync`. Automatic `OPTIONS` requests receive a generated response instead ([app.py:1007](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1007)).

Two related details:

- Registration is setup work: Flask rejects it after the first request has been handled ([scaffold.py:42](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L42), [app.py:411](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L411)).
- On a blueprint, the same `route` decorator calls the blueprint’s implementation, which records registration for later. Registration then applies the URL prefix and prefixes the endpoint with the blueprint name before calling the application’s `add_url_rule` ([blueprints.py:433](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L433), [blueprints.py:98](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L98)).

**Uncertainty:** Flask documents automatic `HEAD` support alongside `GET`, but that behavior is delegated to Werkzeug’s rule implementation. I inspected only this repository, so Werkzeug’s internal implementation was not verified here ([scaffold.py:408](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L408)).

## auto: q16-codeaware

`add_url_rule` registers two related things: a URL rule in `app.url_map`, and an endpoint-to-function mapping in `app.view_functions`. The `route` decorator is a convenience wrapper around it.

For example, these are equivalent:

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

The decorator calls `self.add_url_rule(rule, endpoint, f, **options)` and returns the original function—it does not wrap its execution. See [scaffold.py:368](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L368).

Under the hood, the application implementation does the following:

1. **Chooses the endpoint.** Unless supplied explicitly, it uses `view_func.__name__`—`"hello"` above. The endpoint identifies the view independently of its URL. See [app.py:614](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L614) and [scaffold.py:709](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L709).
2. **Determines HTTP methods.** It uses the `methods` option, then the function’s `methods` attribute, then `GET`. It rejects a bare string, uppercases method names, and includes the function’s `required_methods`. See [app.py:617](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L617).
3. **Configures automatic `OPTIONS`.** An explicit argument takes precedence over the function attribute; otherwise Flask enables it when `OPTIONS` is absent and `PROVIDE_AUTOMATIC_OPTIONS` is enabled. See [app.py:634](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L634).
4. **Registers the rule and callable.** It constructs `self.url_rule_class`—normally Werkzeug’s `Rule`—with the URL, methods, endpoint, and remaining options, then adds it to `url_map`. If a view was supplied, it stores it in `view_functions`. Reusing an endpoint for a different function raises an assertion; reusing it for the same function is allowed. See [app.py:251](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L251) and [app.py:651](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L651).

When a request arrives, URL matching stores the matched rule and extracted arguments on the request ([ctx.py:405](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/ctx.py#L405)). Dispatch then handles automatic `OPTIONS`, or looks up `view_functions[rule.endpoint]` and calls it through `ensure_sync`, passing the URL arguments as keyword arguments ([app.py:1007](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1007)).

Two useful details:

- You can register a rule without a view by supplying an endpoint, then associate its function later using `@app.endpoint(...)` ([scaffold.py:412](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L412)).
- Registration is setup work: Flask rejects it after the application has handled its first request ([app.py:411](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L411)).

**Scope and uncertainty:** This traces the application implementation. Blueprints defer registration and later apply URL and endpoint prefixes ([blueprints.py:433](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L433), [blueprints.py:98](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L98)). Flask documents automatic `HEAD` support ([scaffold.py:408](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L408)), but its underlying Werkzeug implementation was not inspected here.

## auto: q18-baseline

Flask creates the object from the **WSGI environment supplied by the server**, before dispatching the request:

1. The WSGI server calls `app(environ, start_response)`. Flask’s `__call__` forwards to `wsgi_app`, which calls `self.request_context(environ)`. See [src/flask/app.py:1679](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1679) and [src/flask/app.py:1653](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1653).

2. `request_context` calls `AppContext.from_environ(self, environ)`. That method performs the actual construction:
   ```python
   request = app.request_class(environ)
   request.json_module = app.json
   return cls(app, request=request)
   ```
   See [src/flask/app.py:1576](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1576) and [src/flask/ctx.py:339](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/ctx.py#L339).

3. By default, `app.request_class` is Flask’s `Request`, a subclass of Werkzeug’s `Request`. Applications can replace `request_class` with a custom subclass. See [src/flask/app.py:237](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L237) and [src/flask/wrappers.py:18](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/wrappers.py#L18).

4. Flask pushes the context before dispatch. The context is stored in a `ContextVar`, and the exported `flask.request` is a `LocalProxy` that resolves to that context’s request object. See [src/flask/app.py:1657](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1657), [src/flask/ctx.py:433](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/ctx.py#L433), and [src/flask/globals.py:57](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/globals.py#L57).

This checkout uses a combined `AppContext` for application and request information ([src/flask/ctx.py:286](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/ctx.py#L286)). The construction path is explicit; Werkzeug’s internal initialization and parsing were not inspected, so those details are outside this answer.

## auto: q18-codeaware

Flask creates the request object from the **WSGI environment** supplied by the server:

1. The server calls the Flask application’s `__call__(environ, start_response)`, which forwards to `wsgi_app`. ([app.py:1679](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1679))
2. `wsgi_app` calls `self.request_context(environ)`. In this checkout, that delegates to `AppContext.from_environ(self, environ)`. ([app.py:1653](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1653), [app.py:1576](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1576))
3. **The actual construction happens in `from_environ`:**
   ```python
   request = app.request_class(environ)
   request.json_module = app.json
   return cls(app, request=request)
   ```
   This creates a fresh request, assigns the application’s JSON provider, and stores it in the new context. ([ctx.py:346](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/ctx.py#L346))
4. `wsgi_app` pushes that context before dispatching. The context becomes active through a `ContextVar`, and the exported `flask.request` is a `LocalProxy` resolving to its request object. ([app.py:1657](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1657), [ctx.py:433](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/ctx.py#L433), [globals.py:57](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/globals.py#L57))

By default, `app.request_class` is Flask’s `Request`, a subclass of Werkzeug’s `Request`. Applications can replace it with a custom subclass. ([app.py:239](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L239), [wrappers.py:18](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/wrappers.py#L18))

Scope: this traces the normal path in the local checkout. Werkzeug’s constructor internals were not inspected; this checkout uses a combined `AppContext` for app and request data.

## guided: q11-baseline

The blueprint class hierarchy is a single inheritance chain:

```text
object
└── flask.sansio.scaffold.Scaffold
    └── flask.sansio.blueprints.Blueprint
        └── flask.blueprints.Blueprint  (public flask.Blueprint)
```

- **`Scaffold`** provides common behavior shared by applications and blueprints. It declares no explicit superclass, so implicitly inherits `object`. See [scaffold.py:52](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L52).
- **Sans-I/O `Blueprint(Scaffold)`** implements deferred setup and registration on an application. See [sansio/blueprints.py:119](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L119).
- **Public `Blueprint(SansioBlueprint)`** imports that base under the alias `SansioBlueprint` and adds concrete functionality such as CLI setup and static-file serving. See [blueprints.py:10](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/blueprints.py#L10), [blueprints.py:18](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/blueprints.py#L18), and [blueprints.py:45](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/blueprints.py#L45). It is exported as `flask.Blueprint` in [__init__.py:3](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/__init__.py#L3).

The tests demonstrate further subclassing with `MyBlueprint(flask.Blueprint)`, overriding cache behavior: [test_blueprints.py:222](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/tests/test_blueprints.py#L222).

If “tree” means **nested blueprints**, that relationship uses registration: `register_blueprint` stores child blueprint instances in `_blueprints` ([sansio/blueprints.py:256](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L256)).

There is no uncertainty about the declared inheritance chain; dynamically defined external subclasses are outside this source inspection.

## guided: q11-codeaware

The blueprint inheritance chain is:

```text
object
└── flask.sansio.scaffold.Scaffold
    └── flask.sansio.blueprints.Blueprint
        └── flask.blueprints.Blueprint  (public flask.Blueprint)
            └── MyBlueprint            (test-only subclass)
```

- **`Scaffold`** has no explicit base, so it inherits from Python’s `object`. It provides common application and blueprint behavior ([scaffold.py:52](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L52)), including routing decorators ([scaffold.py:344](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L344)).
- **Sans-I/O `Blueprint`** directly inherits `Scaffold` and implements deferred registration of routes and other application functions ([sansio/blueprints.py:119](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L119)).
- **Public `Blueprint`** inherits that class through the alias `SansioBlueprint` ([blueprints.py:10](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/blueprints.py#L10), [blueprints.py:18](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/blueprints.py#L18)). It adds CLI initialization, static-file serving, and resource-file access. This is the class exported as `flask.Blueprint` ([__init__.py:3](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/__init__.py#L3)).
- **`MyBlueprint`** in the tests subclasses the public class to override static-file cache age ([test_blueprints.py:222](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/tests/test_blueprints.py#L222)).

`BlueprintSetupState` is a separate helper class, not a blueprint subclass ([sansio/blueprints.py:34](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L34)).

The core chain is confirmed by source declarations. The static graph also found the test subclass, but its partial indexing cannot establish that no additional subclasses exist.

## guided: q16-baseline

`route` is a convenience decorator around `add_url_rule`. The latter registers the URL pattern and associates its endpoint name with a view function.

These registrations are equivalent:

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

The decorator calls `self.add_url_rule(rule, endpoint, f, **options)` and returns the original function unchanged. It does not wrap the function’s execution. See [scaffold.py:368](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L368).

Under the hood, the application’s `add_url_rule`:

1. **Determines the endpoint.** If omitted, it uses `view_func.__name__`, here `"hello"`. The endpoint is the key linking a routing rule to its callable. See [app.py:614](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L614) and [scaffold.py:709](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L709).
2. **Resolves HTTP methods.** It uses explicit `methods`, otherwise `view_func.methods`, otherwise `GET`; rejects a plain string; uppercases method names; and includes `view_func.required_methods`. Automatic `OPTIONS` depends on the explicit argument, the function’s `provide_automatic_options` attribute, and application configuration. See [app.py:617](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L617).
3. **Creates and stores the routing rule.** It constructs `self.url_rule_class`—normally Werkzeug’s `Rule`—with the URL, endpoint, methods, and remaining options, then adds it to `self.url_map`. See [app.py:651](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L651) and [app.py:251](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L251).
4. **Registers the callable.** It stores `self.view_functions[endpoint] = view_func`. Reusing an endpoint for a different function raises `AssertionError`; multiple rules may share the same endpoint and function. See [app.py:655](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L655).

When a request arrives, the URL adapter matches a rule and extracts URL arguments ([ctx.py:405](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/ctx.py#L405)). Dispatch then looks up `view_functions[rule.endpoint]` and invokes it through `ensure_sync`, passing those arguments as keyword arguments. Automatic `OPTIONS` is handled before invoking the view. See [app.py:1007](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1007).

Two useful details:

- You can register a rule with an explicit endpoint but no view, then bind the function later using `@app.endpoint(...)`: [scaffold.py:412](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L412).
- Blueprint routes defer registration; when applied to an app, they add the URL prefix and qualify the endpoint with the blueprint name: [blueprints.py:433](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L433), [blueprints.py:98](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L98).

**Scope and uncertainty:** This describes the checked-out implementation. Werkzeug performs the actual matching and automatic `HEAD` handling for `GET` rules; its internal implementation was not inspected. Flask documents that automatic behavior in [scaffold.py:408](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L408).

## guided: q16-codeaware

`route` is a convenience decorator around `add_url_rule`. It registers the function when the decorator runs, then returns the same function unchanged—it does not wrap request execution. The implementation extracts `endpoint` and calls `self.add_url_rule(rule, endpoint, f, **options)` ([scaffold.py:368](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L368)).

These registrations are equivalent:

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

Under the hood, the application’s `add_url_rule`:

1. **Chooses an endpoint**, the string identifying the view. Unless explicitly supplied, it uses `view_func.__name__` ([app.py:614](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L614), [scaffold.py:709](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L709)).
2. **Determines HTTP methods.** It uses the supplied `methods`, otherwise the function’s `methods` attribute, otherwise `GET`. It rejects a bare string, uppercases method names, and includes the function’s `required_methods` ([app.py:617](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L617)).
3. **Configures automatic `OPTIONS`.** An explicit argument takes precedence over the function attribute; otherwise Flask enables it when `OPTIONS` is absent and `PROVIDE_AUTOMATIC_OPTIONS` is enabled ([app.py:634](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L634)).
4. **Registers two linked pieces:** a routing rule in `app.url_map`, and an endpoint-to-callable entry in `app.view_functions`. The rule class defaults to Werkzeug’s `Rule`; extra options go to its constructor ([app.py:251](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L251), [app.py:651](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L651)). Reusing an endpoint for a different function raises an assertion; multiple rules can share the same function ([app.py:655](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L655)).

During a request, the URL adapter matches a rule and extracts URL arguments ([ctx.py:405](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/ctx.py#L405)). Dispatch then either generates the automatic `OPTIONS` response or calls `view_functions[rule.endpoint]` with those arguments through `ensure_sync` ([app.py:1007](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1007)). Registration itself never calls the view.

Both registration APIs are setup methods: application registration is rejected after the first request ([scaffold.py:42](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L42), [app.py:411](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L411)).

For **blueprints**, the same decorator calls the blueprint’s `add_url_rule`, which records deferred registration. When registered on an app, it applies URL and endpoint prefixes and forwards to the app’s implementation ([blueprints.py:433](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L433), [blueprints.py:334](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L334), [blueprints.py:98](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L98)).

**Scope and uncertainty:** Flask documents automatic `HEAD` support ([scaffold.py:408](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L408)), but its underlying Werkzeug implementation is outside this repository and was not inspected.

## guided: q18-baseline

Flask creates the `Request` from the WSGI environment supplied by the HTTP server, before dispatching to a view:

1. The server calls `Flask.__call__(environ, start_response)`, which forwards to `wsgi_app()`. See [src/flask/app.py:1679](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1679).
2. `wsgi_app()` calls `self.request_context(environ)`, which delegates to `AppContext.from_environ(self, environ)`. See [src/flask/app.py:1653](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1653) and [src/flask/app.py:1576](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1576).
3. **The actual creation happens in `AppContext.from_environ()`:**
   ```python
   request = app.request_class(environ)
   request.json_module = app.json
   return cls(app, request=request)
   ```
   See [src/flask/ctx.py:346](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/ctx.py#L346). By default, `request_class` is Flask’s `Request`, a subclass of Werkzeug’s request wrapper; applications can replace it. See [src/flask/app.py:239](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L239) and [src/flask/wrappers.py:18](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/wrappers.py#L18).
4. Flask pushes the context before dispatching. Pushing binds it to a `ContextVar`, and the exported `flask.request` proxy resolves to that context’s request object. See [src/flask/app.py:1657](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1657), [src/flask/ctx.py:433](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/ctx.py#L433), and [src/flask/globals.py:57](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/globals.py#L57).

This checkout uses the combined `AppContext` implementation, which documents its merger with `RequestContext` in Flask 3.2 ([src/flask/ctx.py:286](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/ctx.py#L286)). The default creation path is explicit; Werkzeug’s constructor internals and application-specific overrides were not inspected.

## guided: q18-codeaware

Flask creates the object from the incoming **WSGI environment**, using `app.request_class(environ)`.

In this checkout, the flow is:

1. The WSGI server calls `Flask.__call__(environ, start_response)`, which delegates to `wsgi_app`. [src/flask/app.py:1679](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1679)
2. `wsgi_app` calls `self.request_context(environ)`. That method delegates to `AppContext.from_environ(self, environ)`. [src/flask/app.py:1653](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1653), [src/flask/app.py:1576](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1576)
3. `AppContext.from_environ` constructs the request, assigns the app’s JSON provider, and puts it in a new context:
   ```python
   request = app.request_class(environ)
   request.json_module = app.json
   return cls(app, request=request)
   ```
   [src/flask/ctx.py:346](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/ctx.py#L346)
4. `wsgi_app` pushes that context before dispatching the request. The exported `flask.request` is a `LocalProxy` resolving to the active context’s request object. [src/flask/app.py:1657](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L1657), [src/flask/globals.py:57](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/globals.py#L57)

By default, `request_class` is Flask’s `Request`, a subclass of Werkzeug’s `Request`; applications can replace it with a custom subclass. [src/flask/app.py:239](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/app.py#L239), [src/flask/wrappers.py:18](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/wrappers.py#L18)

The default creation path is explicit in the source. Werkzeug’s internal initialization was not inspected, and application overrides could change this behavior.
