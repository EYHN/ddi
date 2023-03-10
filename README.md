# DDI (dynamic dependency injection)

[![](https://img.shields.io/crates/v/ddi)](https://crates.io/crates/ddi) ![](https://img.shields.io/crates/l/ddi)

This library provides a generic dependency injection container that can be easily integrated into any application and can be extremely extensible with the [extension trait](https://rust-lang.github.io/rfcs/0445-extension-trait-conventions.html).

Dependency injection is a common design pattern , mainly used in some frameworks such as [Rocket](https://rocket.rs/), [Actix Web](https://actix.rs/), [bevy](https://bevyengine.org/). With `ddi` you can implement dependency injection without such frameworks, and you can implement your own framework.

> scope feature (singleton vs transient vs scoped/request) is on the way!

## Example

```rust
use ddi::*;

struct TestService(String);

let mut services = ServiceCollection::new();
services.service(1usize);
services.service("helloworld");
services.service_factory(|num: &usize, str: &&str| Ok(TestService(format!("{}{}", num, str))));

let provider = services.provider();
assert_eq!(provider.get::<TestService>().unwrap().0, "1helloworld");
```

## `std` feature

`ddi` supports `no-std` by default, if `std` feature enabled the internal data structure will be changed from [`alloc::collections::BTreeMap`] to [`std::collections::HashMap`] and [`std::error::Error`] will be implemented for [`DDIError`]. This will give a little performance improvement and usability.

## `sync` feature

If `sync` feature enabled, `ddi` will support multi-threading and you can share [`ServiceProvider`] between multiple threads.

>! Enabling `sync` may cause your existing code to not compile! This is because enabling `sync` requires instances in the [`ServiceCollection`] to implement `send + sync` and `ServiceFactory` to implement `send`. And default no such restrictions.

## Basic Usage

First you need to register all services in the [`ServiceCollection`], which is a container of all services, [`ServiceCollection`] stored a series of triplets (type, name, implementation). You can use the [`ServiceCollection::service`] to add item to it.

For example, the following code will add a item (&str, "default", "helloworld") to the [`ServiceCollection`]

```rust ignore
let mut services = ServiceCollection::new();
services.service("helloworld");
```

Here, the service "implementation" can also be a function, the factory of the service. The factory function is lazy execution, will only be executed when the service is used. For example.

```rust ignore
services.service_factory(|| Ok("helloworld"));
```

The service factory can use parameters to get other services as dependencies. `ddi` will pass in the corresponding services based on the type of the parameters. Due to the reference rule of rust, the type of the parameter must be an immutable reference type.

```rust ignore
services.service_factory(|dep: &Foo| Ok(Bar::new(dep)));
```

When you have all the services registered, use [`ServiceCollection::provider()`] to get the [`ServiceProvider`], and then you can get any service you want from [`ServiceProvider`].

```rust ignore
let provider = services.provider();
assert_eq!(provider.get::<TestService>().unwrap().0, "helloworld");
```

## Design Patterns

#### \* Wrap your service with [`Service<T>`] (Arc)

When a service wants to hold references to other services, the referenced service should be wrapped in [`Arc<T>`] for proper lifecycle handling. `ddi` defines an alias `type Service<T> = Arc<T>` for such a pattern.

We recommend wrapping all services in [`Service<T>`] to make cross-references between services easier.

That does not allow circular references, because `ddi` does not allow circular dependencies, which would cause the [`DDIError::CircularDependencyDetected`] error.

```rust
use ddi::*;

struct Bar;
struct Foo(Service<Bar>);

let mut services = ServiceCollection::new();
services.service(Service::new(Bar));
services.service_factory(
  |bar: &Service<Bar>| Ok(Service::new(Foo(bar.clone())))
);

let provider = services.provider();
assert!(provider.get::<Service<Foo>>().is_ok());
```

#### \* Use extension trait to expanding [`ServiceCollection`]

The extension trait makes [`ServiceCollection`] extremely extensible. The following example shows the use of the extension trait to register multiple services into one function.

```rust
use ddi::*;

// ------------ definition ------------

#[derive(Clone)]
struct DbConfiguration;
struct DbService(DbConfiguration, Service<DbConnectionManager>);
struct DbConnectionManager;

pub trait DbServiceCollectionExt: ServiceCollectionExt {
    fn install_database(&mut self) {
      self.service(Service::new(DbConnectionManager));
      self.service_factory(
        |config: &DbConfiguration, cm: &Service<DbConnectionManager>|
          Ok(Service::new(DbService(config.clone(), cm.clone())))
      );
      self.service(DbConfiguration);
    }
}

impl<T: ServiceCollectionExt> DbServiceCollectionExt for T {}

// -------------- usage ---------------

let mut services = ServiceCollection::new();

services.install_database();

let provider = services.provider();
assert!(provider.get::<Service<DbService>>().is_ok());
```

#### \* Use [`ServiceProvider`] in the factory, get other services dynamically

In our previous examples service factory used static parameters to get the dependencies, in the following example we use [`ServiceProvider`] to get the dependencies dynamically.

```rust
use ddi::*;

trait Decoder: Send + Sync { fn name(&self) -> &'static str; }
struct HardwareDecoder;
struct SoftwareDecoder;
impl Decoder for HardwareDecoder { fn name(&self) -> &'static str { "hardware" } }
impl Decoder for SoftwareDecoder { fn name(&self) -> &'static str { "software" } }
struct Playback {
  decoder: Service<dyn Decoder>
}

const SUPPORT_HARDWARE_DECODER: bool = false;

let mut services = ServiceCollection::new();

if SUPPORT_HARDWARE_DECODER {
  services.service(Service::new(HardwareDecoder));
}
services.service(Service::new(SoftwareDecoder));
services.service_factory(
  |provider: &ServiceProvider| {
    if let Ok(hardware) = provider.get::<Service<HardwareDecoder>>() {
      Ok(Playback { decoder: hardware.clone() })
    } else {
      Ok(Playback { decoder: provider.get::<Service<SoftwareDecoder>>()?.clone() })
    }
  }
);

let provider = services.provider();
assert_eq!(provider.get::<Playback>().unwrap().decoder.name(), "software");
```

#### \* Use `service_var` or `service_factory_var` to register variants of service

The [`ServiceCollection`] can register multiple variants of the same type of service, using `service_var` or `service_factory_var`. When registering variants you need to declare ServiceName for each variant, the default (registered using the service or service_factory function) ServiceName is "default".

The following example demonstrates how to build an http server based on service variants.

#[doc(cfg(not(feature = "sync")))]

```rust
use ddi::*;

type Route = Service<dyn Fn() -> String>;
struct HttpService {
  routes: std::collections::HashMap<String, Route>
}
struct BusinessService {
  value: String
}

let mut services = ServiceCollection::new();

services.service_var("/index", Service::new(|| "<html>".to_string()) as Route);
services.service_var("/404", Service::new(|| "404".to_string()) as Route);
services.service_factory_var(
  "/business",
  |business: &Service<BusinessService>| {
    let owned_business = business.clone();
    Ok(Service::new(move || owned_business.value.clone()) as Route)
  }
);
services.service_factory(
  |provider: &ServiceProvider| {
    let routes = provider.get_all::<Route>()?
      .into_iter()
      .map(|(path, route)| (path.to_string(), route.clone()))
      .collect();
    Ok(HttpService { routes })
  }
);
services.service(Service::new(BusinessService {
  value: "hello".to_string()
}));

let provider = services.provider();
assert_eq!(provider.get::<HttpService>().unwrap().routes.get("/index").unwrap()(), "<html>");
assert_eq!(provider.get::<HttpService>().unwrap().routes.get("/404").unwrap()(), "404");
assert_eq!(provider.get::<HttpService>().unwrap().routes.get("/business").unwrap()(), "hello");
```

#### \* Use extension trait to simplify the registration of service variants

In the previous example we used `service_var` and `service_factory_var` to register routes for the http server, but the code were obscure and no type checking. The following example demonstrates use extension trait to simplify the definition of routes and solve these problems.

```rust
use ddi::*;

// ------------ definition ------------

type Route = Service<dyn Fn() -> String>;
struct HttpService {
  routes: std::collections::HashMap<String, Route>
}
struct BusinessService {
  value: String
}

pub trait HttpCollectionExt: ServiceCollectionExt {
    fn install_http(&mut self) {
      self.service_factory(
        |provider: &ServiceProvider| {
          let routes = provider.get_all::<Route>()?
            .into_iter()
            .map(|(path, route)| (path.to_string(), route.clone()))
            .collect();
          Ok(HttpService { routes })
        }
      );
    }

    fn install_route<Param>(&mut self, path: &'static str, route: impl ServiceFn<Param, String> + 'static) {
      self.service_factory_var(path, move |provider: &ServiceProvider| {
        let owned_provider = provider.clone();
        Ok(Service::new(move || route.run_with(&owned_provider).expect("123")) as Route)
      })
    }
}

impl<T: ServiceCollectionExt> HttpCollectionExt for T {}

// -------------- usage ---------------

let mut services = ServiceCollection::new();

services.install_route("/index", || "<html>".to_string());
services.install_route("/404", || "404".to_string());
services.install_route("/business", |business: &BusinessService| business.value.to_string());
services.install_http();

services.service(BusinessService {
  value: "hello".to_string()
});

let provider = services.provider();
assert_eq!(provider.get::<HttpService>().unwrap().routes.get("/index").unwrap()(), "<html>");
assert_eq!(provider.get::<HttpService>().unwrap().routes.get("/404").unwrap()(), "404");
assert_eq!(provider.get::<HttpService>().unwrap().routes.get("/business").unwrap()(), "hello");
```

The `install_route` function in the example uses the [`ServiceFn`] trait as argument, which is a powerful type (of course we have [`ServiceFnMut`] and [`ServiceFnOnce`]), using the [`ServiceFn::run_with`] function to automatically extract Fn arguments from the [`ServiceProvider`] and execute it.

# License

This project is licensed under [The MIT License](https://github.com/EYHN/ddi/blob/main/LICENSE).

# Credits

Inspired by [Dependency injection in .NET](https://learn.microsoft.com/en-us/dotnet/core/extensions/dependency-injection).
