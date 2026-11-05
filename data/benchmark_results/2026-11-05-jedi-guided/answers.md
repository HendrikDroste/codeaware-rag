# Q11, Q17 and Q18 answers

Raw answers copied unchanged.

## Q11, baseline

The blueprint class has a single inheritance chain:

```text
object
└── flask.sansio.scaffold.Scaffold
    └── flask.sansio.blueprints.Blueprint
        └── flask.blueprints.Blueprint  (public flask.Blueprint)
```

- **`Scaffold`** supplies shared application/blueprint behavior, including route decorators, request hooks, and error handlers. It declares no explicit base, so it inherits from Python’s `object`. See [scaffold.py:52](/Users/hendrikdroste/PycharmProjects/flask/src/flask/sansio/scaffold.py:52).
- **Sans-IO `Blueprint`** inherits `Scaffold` and provides blueprint registration and deferred setup behavior. See [sansio/blueprints.py:119](/Users/hendrikdroste/PycharmProjects/flask/src/flask/sansio/blueprints.py:119).
- **Public `Blueprint`** inherits that class under the alias `SansioBlueprint`, adding concrete file-serving and resource-opening methods. See the import at [blueprints.py:10](/Users/hendrikdroste/PycharmProjects/flask/src/flask/blueprints.py:10) and class declaration at [blueprints.py:18](/Users/hendrikdroste/PycharmProjects/flask/src/flask/blueprints.py:18). It is exported as `flask.Blueprint` at [__init__.py:3](/Users/hendrikdroste/PycharmProjects/flask/src/flask/__init__.py:3).

If you meant **nested blueprints**, that is composition: `register_blueprint` stores child blueprints, rather than creating subclasses ([sansio/blueprints.py:256–271](/Users/hendrikdroste/PycharmProjects/flask/src/flask/sansio/blueprints.py:256)).

There is no uncertainty about the declared inheritance chain; the only ambiguity is whether “tree” meant class inheritance or blueprint nesting.

## Q11, codeaware

The blueprint inheritance chain is:

```text
object
└── flask.sansio.scaffold.Scaffold
    └── flask.sansio.blueprints.Blueprint
        └── flask.blueprints.Blueprint
```

- **`Scaffold`** has no explicit base, so it inherits Python’s `object`. It provides behavior shared by Flask applications and blueprints: [src/flask/sansio/scaffold.py:52](/Users/hendrikdroste/PycharmProjects/flask/src/flask/sansio/scaffold.py:52).
- **The Sans-IO `Blueprint`** directly inherits `Scaffold`. It represents routes and application functions recorded for later registration: [src/flask/sansio/blueprints.py:119](/Users/hendrikdroste/PycharmProjects/flask/src/flask/sansio/blueprints.py:119).
- **The concrete `Blueprint`** inherits that Sans-IO class, named `SansioBlueprint` in its declaration: [src/flask/blueprints.py:18](/Users/hendrikdroste/PycharmProjects/flask/src/flask/blueprints.py:18).

There is no uncertainty about this base-class chain. This describes the repository’s blueprint implementation; it does not claim to enumerate every custom or runtime-defined subclass.

## Q17, baseline

Assuming you mean configuration defaults, `App` loads them during initialization:

- `App.__init__` calls `self.make_config(instance_relative_config)` at [src/flask/sansio/app.py:317](/Users/hendrikdroste/PycharmProjects/flask/src/flask/sansio/app.py:317).
- `make_config()` copies `self.default_config`, overrides `DEBUG` using `get_debug_flag()`, and passes the defaults into the configuration class ([lines 480–494](/Users/hendrikdroste/PycharmProjects/flask/src/flask/sansio/app.py:480)).
- The actual default values are defined in `Flask.default_config` at [src/flask/app.py:205–235](/Users/hendrikdroste/PycharmProjects/flask/src/flask/app.py:205).

`DEBUG` comes from the `FLASK_DEBUG` environment variable and defaults to `False` ([src/flask/helpers.py:28–33](/Users/hendrikdroste/PycharmProjects/flask/src/flask/helpers.py:28)).

## Q17, codeaware

Assuming you mean **configuration defaults**, the app loads them during initialization:

- `App.__init__` sets `self.config = self.make_config(instance_relative_config)` in [src/flask/sansio/app.py:317](/Users/hendrikdroste/PycharmProjects/flask/src/flask/sansio/app.py:317).
- `make_config()` copies `self.default_config`, overrides `DEBUG` using `get_debug_flag()`, and passes the defaults to the configuration constructor in [src/flask/sansio/app.py:492–494](/Users/hendrikdroste/PycharmProjects/flask/src/flask/sansio/app.py:492).
- The actual default values are defined in `Flask.default_config` in [src/flask/app.py:205–235](/Users/hendrikdroste/PycharmProjects/flask/src/flask/app.py:205), including `TESTING=False` and `SESSION_COOKIE_NAME="session"`.

The only uncertainty is whether “default values” refers to configuration or another kind of default.

## Q18, baseline

Flask creates the request object from the **WSGI environment** supplied by the server:

1. The server calls `app(environ, start_response)`. Flask’s `__call__` forwards this to `wsgi_app`. [src/flask/app.py:1679](src/flask/app.py#L1679)
2. `wsgi_app` calls `self.request_context(environ)`, which delegates to `AppContext.from_environ(self, environ)`. [src/flask/app.py:1653](src/flask/app.py#L1653), [src/flask/app.py:1576](src/flask/app.py#L1576)
3. **The actual construction is `request = app.request_class(environ)`.** Flask then assigns the app’s JSON provider to the request and stores the request in a new context. [src/flask/ctx.py:346](src/flask/ctx.py#L346)
4. `request_class` defaults to Flask’s `Request`, a subclass of Werkzeug’s request wrapper. Applications can replace it with a custom class. [src/flask/app.py:237](src/flask/app.py#L237), [src/flask/wrappers.py:18](src/flask/wrappers.py#L18)
5. Flask pushes the context before dispatching the request. This sets the active context variable; `flask.request` is a `LocalProxy` that resolves to that context’s request object. The context is popped afterward. [src/flask/ctx.py:433](src/flask/ctx.py#L433), [src/flask/globals.py:57](src/flask/globals.py#L57), [src/flask/app.py:1657](src/flask/app.py#L1657), [src/flask/app.py:1677](src/flask/app.py#L1677)

This checkout uses a combined `AppContext` for request handling; its documentation says `RequestContext` was merged into it in version 3.2. No uncertainty about this construction path; Werkzeug’s internal parsing behavior was not inspected. [src/flask/ctx.py:286](src/flask/ctx.py#L286)

## Q18, codeaware

Flask creates the request object from the **WSGI environment supplied by the server**, before dispatching to a view:

1. The server calls the Flask application with `environ` and `start_response`. `Flask.__call__` forwards these to `wsgi_app`. ([app.py:1679](src/flask/app.py#L1679))
2. `wsgi_app` calls `self.request_context(environ)`, which delegates to `AppContext.from_environ(self, environ)`. ([app.py:1653](src/flask/app.py#L1653), [app.py:1562](src/flask/app.py#L1562))
3. **The actual construction is `request = app.request_class(environ)`.** Flask then sets `request.json_module = app.json` and stores the request in a new context. By default, `request_class` is Flask’s `Request`, a Werkzeug `Request` subclass; applications can replace it. ([ctx.py:340](src/flask/ctx.py#L340), [app.py:239](src/flask/app.py#L239), [wrappers.py:18](src/flask/wrappers.py#L18))
4. `wsgi_app` pushes the context and dispatches the request. The `flask.request` global is a `LocalProxy` to the active context’s request object. The context is popped during cleanup. ([app.py:1653](src/flask/app.py#L1653), [globals.py:57](src/flask/globals.py#L57), [app.py:1677](src/flask/app.py#L1677))

This describes the default path in this checkout. Custom request classes or overridden request-handling methods can change the behavior.
