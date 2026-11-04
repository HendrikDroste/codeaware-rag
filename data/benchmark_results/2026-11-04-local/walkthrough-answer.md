`route()` is a convenience decorator around `add_url_rule()`. The latter registers routing metadata and, optionally, the view function; it does not execute the view.

The class relationship is `Flask → App → Scaffold`. The public package exports `Flask`, which inherits `App`; `App` inherits `Scaffold`. `Scaffold` supplies `route()` and an `add_url_rule()` stub, while `App` supplies the concrete registration implementation. See [__init__.py:2](/Users/hendrikdroste/PycharmProjects/flask/src/flask/__init__.py:2), [app.py:108](/Users/hendrikdroste/PycharmProjects/flask/src/flask/app.py:108), [sansio/app.py:60](/Users/hendrikdroste/PycharmProjects/flask/src/flask/sansio/app.py:60), and [scaffold.py:441](/Users/hendrikdroste/PycharmProjects/flask/src/flask/sansio/scaffold.py:441).

Under the hood, `App.add_url_rule()`:

1. **Checks that setup is still allowed.** Its `@setupmethod` wrapper checks application setup state; registration is rejected after the first request. [scaffold.py:42](/Users/hendrikdroste/PycharmProjects/flask/src/flask/sansio/scaffold.py:42), [sansio/app.py:411](/Users/hendrikdroste/PycharmProjects/flask/src/flask/sansio/app.py:411).
2. **Chooses an endpoint and methods.** The endpoint defaults to `view_func.__name__`. Methods come from the explicit option, then `view_func.methods`, then `GET`. A bare string is rejected, and method names are uppercased. [sansio/app.py:614](/Users/hendrikdroste/PycharmProjects/flask/src/flask/sansio/app.py:614), [scaffold.py:709](/Users/hendrikdroste/PycharmProjects/flask/src/flask/sansio/scaffold.py:709).
3. **Handles required methods and automatic `OPTIONS`.** It incorporates `view_func.required_methods`. Automatic options follows the explicit argument, then the view attribute, then configuration when `OPTIONS` is absent. [sansio/app.py:631](/Users/hendrikdroste/PycharmProjects/flask/src/flask/sansio/app.py:631).
4. **Creates and stores the rule.** It constructs `self.url_rule_class(...)`, records the automatic-options flag, and adds it to `self.url_map`. These classes default to Werkzeug’s `Rule` and `Map`. [sansio/app.py:651](/Users/hendrikdroste/PycharmProjects/flask/src/flask/sansio/app.py:651), [sansio/app.py:251](/Users/hendrikdroste/PycharmProjects/flask/src/flask/sansio/app.py:251).
5. **Associates the endpoint with the view.** If provided, the function goes into `self.view_functions[endpoint]`. Mapping an existing endpoint to a different function raises an assertion. An explicit endpoint allows registration without a view. [sansio/app.py:655](/Users/hendrikdroste/PycharmProjects/flask/src/flask/sansio/app.py:655).

The decorator’s entire registration step is effectively:

```python
endpoint = options.pop("endpoint", None)
self.add_url_rule(rule, endpoint, f, **options)
return f
```

Thus `@app.route("/", methods=["GET"])` registers the decorated function through the same machinery and returns that function unchanged. Python’s method lookup selects `App.add_url_rule()` on a `Flask` instance. [scaffold.py:368](/Users/hendrikdroste/PycharmProjects/flask/src/flask/sansio/scaffold.py:368).

Later, request dispatch uses the matched rule’s endpoint to retrieve and call `view_functions[rule.endpoint]`, passing URL arguments. Automatic `OPTIONS` takes a separate response path. [app.py:1011](/Users/hendrikdroste/PycharmProjects/flask/src/flask/app.py:1011).

**Graph evidence and uncertainty:** Codeaware reported a fresh but partial index. Its inheritance results established the class chain. Callers included static-file registration and blueprint registration; callees resolved only `_endpoint_from_view_func`. It did **not** return the decorator-to-`App.add_url_rule` edge or the rule-construction/map-add edges, so those conclusions required source inspection. Missing static edges do not establish absent runtime calls. The repository documents automatic `HEAD` support, but its underlying Werkzeug implementation was outside this inspection. [app.py:356](/Users/hendrikdroste/PycharmProjects/flask/src/flask/app.py:356), [blueprints.py:110](/Users/hendrikdroste/PycharmProjects/flask/src/flask/sansio/blueprints.py:110), [scaffold.py:360](/Users/hendrikdroste/PycharmProjects/flask/src/flask/sansio/scaffold.py:360).