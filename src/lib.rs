#![doc = include_str!("../README.md")]

use std::{
    any::{type_name, Any, TypeId},
    cell::{RefCell, UnsafeCell},
    collections::HashMap,
    error,
    fmt::{self, Debug, Display},
    hash::Hash,
    sync::Arc,
};

pub type Service<T> = Arc<T>;

#[derive(Clone, Copy)]
pub enum ServiceSymbol {
    Type(TypeId, /* debug message */ &'static str),
}

impl ServiceSymbol {
    pub fn new<T: 'static>() -> Self {
        Self::Type(TypeId::of::<T>(), type_name::<T>())
    }
}

impl Debug for ServiceSymbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Type(_, type_name) => write!(f, "{}", type_name),
        }
    }
}

impl Hash for ServiceSymbol {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::Type(type_id, _) => type_id.hash(state),
        }
    }
}

impl PartialEq for ServiceSymbol {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Type(type_id, _), Self::Type(type_id_other, _)) => type_id == type_id_other,
        }
    }
}

impl Eq for ServiceSymbol {}

#[derive(Clone)]
pub enum ServiceName {
    Static(&'static str),
    Dynamic(Arc<String>),
}

impl ServiceName {
    pub fn name(&self) -> &str {
        match self {
            ServiceName::Static(s) => s,
            ServiceName::Dynamic(s) => s,
        }
    }
}

impl Hash for ServiceName {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let name = match self {
            ServiceName::Static(s) => *s,
            ServiceName::Dynamic(s) => s,
        };
        state.write(name.as_bytes());
        state.write_u8(0xff);
    }
}

impl Debug for ServiceName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        std::fmt::Debug::fmt(self.name(), f)
    }
}

impl Display for ServiceName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        std::fmt::Display::fmt(self.name(), f)
    }
}

impl PartialEq for ServiceName {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl Eq for ServiceName {}

impl From<&'static str> for ServiceName {
    fn from(s: &'static str) -> Self {
        ServiceName::Static(s)
    }
}

impl From<String> for ServiceName {
    fn from(s: String) -> Self {
        ServiceName::Dynamic(Arc::new(s))
    }
}

pub trait ServiceFnOnce<Param, Out> {
    fn run_once(self, dependencies: &[&dyn Any]) -> Out;
    fn dependencies() -> Vec<(ServiceSymbol, ServiceName)>;
    fn run_with_once(self, service_ref: &dyn ServiceResolver) -> DDIResult<Out>;
}

pub trait ServiceFnMut<Param, Out>: ServiceFnOnce<Param, Out> {
    fn run_mut(&mut self, dependencies: &[&dyn Any]) -> Out;
    fn run_with_mut(&mut self, service_ref: &dyn ServiceResolver) -> DDIResult<Out>;
}

pub trait ServiceFn<Param, Out>: ServiceFnMut<Param, Out> {
    fn run(&self, dependencies: &[&dyn Any]) -> Out;
    fn run_with(&self, service_ref: &dyn ServiceResolver) -> DDIResult<Out>;
}

macro_rules! impl_service_function {
    ($($param: ident),*) => {
        #[allow(non_snake_case)]
        impl<Out: 'static, Func, $($param: 'static),*> ServiceFn<($($param,)*), Out>
            for Func
        where
            Func: Fn($(&$param,)*) -> Out,
        {
            fn run(&self, dependencies: &[&dyn Any]) -> Out {
                fn call_inner<Out, $($param,)*>(f: impl Fn($(&$param,)*) -> Out, $($param: &$param,)*) -> Out {
                    f($($param,)*)
                }
                if let [$($param,)*] = dependencies {
                    $(let $param = $param.downcast_ref::<$param>().expect(&format!("Failed downcast to {}", std::any::type_name::<$param>()));)*
                    call_inner(self, $($param,)*)
                } else {
                    unreachable!()
                }
            }

            fn run_with(&self, service: &dyn ServiceResolver) -> DDIResult<Out> {
                let dependencies = Self::dependencies();
                let service_ref = ServiceRef {
                    // SAFETY: Here use `core::mem::transmute` to modify the life cycle of `&dyn ServiceResolver` to 'static, because `service_ref` is only lives in this function so it is safe.
                    resolver: unsafe { core::mem::transmute(service as &dyn ServiceResolver) },
                };
                let mut deps = Vec::with_capacity(dependencies.len());
                for (dep_symbol, dep_name) in dependencies.into_iter() {
                    let value: &dyn Any = if dep_symbol == ServiceSymbol::new::<ServiceRef>() {
                        &service_ref
                    } else
                    {
                        service.resolve(dep_symbol.clone(), dep_name)?
                    };

                    deps.push(value);
                }

                Ok(self.run(&deps))
            }
        }

        #[allow(non_snake_case)]
        impl<Out: 'static, Func, $($param: 'static),*> ServiceFnMut<($($param,)*), Out>
            for Func
        where
            Func: FnMut($(&$param,)*) -> Out,
        {
            fn run_mut(&mut self, dependencies: &[&dyn Any]) -> Out {
                fn call_inner<Out, $($param,)*>(mut f: impl FnMut($(&$param,)*) -> Out, $($param: &$param,)*) -> Out {
                    f($($param,)*)
                }
                if let [$($param,)*] = dependencies {
                    $(let $param = $param.downcast_ref::<$param>().expect(&format!("Failed downcast to {}", std::any::type_name::<$param>()));)*
                    call_inner(self, $($param,)*)
                } else {
                    unreachable!()
                }
            }

            fn run_with_mut(&mut self, service: &dyn ServiceResolver) -> DDIResult<Out> {
                let dependencies = Self::dependencies();
                let service_ref = ServiceRef {
                    // SAFETY: Here use `core::mem::transmute` to modify the life cycle of `&dyn ServiceResolver` to 'static, because `service_ref` is only lives in this function so it is safe.
                    resolver: unsafe { core::mem::transmute(service as &dyn ServiceResolver) },
                };
                let mut deps = Vec::with_capacity(dependencies.len());
                for (dep_symbol, dep_name) in dependencies.into_iter() {
                    let value: &dyn Any = if dep_symbol == ServiceSymbol::new::<ServiceRef>() {
                        &service_ref
                    } else
                    {
                        service.resolve(dep_symbol.clone(), dep_name)?
                    };

                    deps.push(value);
                }

                Ok(self.run_mut(&deps))
            }
        }

        #[allow(non_snake_case)]
        impl<Out: 'static, Func, $($param: 'static),*> ServiceFnOnce<($($param,)*), Out>
            for Func
        where
            Func: FnOnce($(&$param,)*) -> Out,
        {
            fn run_once(self, dependencies: &[&dyn Any]) -> Out {
                fn call_inner<Out, $($param,)*>(f: impl FnOnce($(&$param,)*) -> Out, $($param: &$param,)*) -> Out {
                    f($($param,)*)
                }
                if let [$($param,)*] = dependencies {
                    $(let $param = $param.downcast_ref::<$param>().expect(&format!("Failed downcast to {}", std::any::type_name::<$param>()));)*
                    call_inner(self, $($param,)*)
                } else {
                    unreachable!()
                }
            }

            fn dependencies() -> Vec<(ServiceSymbol, ServiceName)> {
                vec![
                    $((ServiceSymbol::new::<$param>(), ServiceName::from("default")),)*
                ]
            }

            fn run_with_once(self, service: &dyn ServiceResolver) -> DDIResult<Out> {
                let dependencies = Self::dependencies();
                let service_ref = ServiceRef {
                    // SAFETY: Here use `core::mem::transmute` to modify the life cycle of `&dyn ServiceResolver` to 'static, because `service_ref` is only lives in this function so it is safe.
                    resolver: unsafe { core::mem::transmute(service as &dyn ServiceResolver) },
                };
                let mut deps = Vec::with_capacity(dependencies.len());
                for (dep_symbol, dep_name) in dependencies.into_iter() {
                    let value: &dyn Any = if dep_symbol == ServiceSymbol::new::<ServiceRef>() {
                        &service_ref
                    } else
                    {
                        service.resolve(dep_symbol, dep_name)?
                    };

                    deps.push(value);
                }

                Ok(self.run_once(&deps))
            }
        }
    };
}

impl_service_function!();
impl_service_function!(A);
impl_service_function!(A, B);
impl_service_function!(A, B, C);
impl_service_function!(A, B, C, D);
impl_service_function!(A, B, C, D, E);
impl_service_function!(A, B, C, D, E, F);
impl_service_function!(A, B, C, D, E, F, G);
impl_service_function!(A, B, C, D, E, F, G, H);
impl_service_function!(A, B, C, D, E, F, G, H, I);
impl_service_function!(A, B, C, D, E, F, G, H, I, J);
impl_service_function!(A, B, C, D, E, F, G, H, I, J, K);
impl_service_function!(A, B, C, D, E, F, G, H, I, J, K, L);
impl_service_function!(A, B, C, D, E, F, G, H, I, J, K, L, M);
impl_service_function!(A, B, C, D, E, F, G, H, I, J, K, L, M, N);
impl_service_function!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O);
impl_service_function!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);

enum ServiceFactory {
    FnOnce(
        Option<
            Box<
                dyn FnOnce(&dyn ServiceResolver) -> DDIResult<Box<dyn Any + Send>> + 'static + Send,
            >,
        >,
    ),
    FnMut(Box<dyn FnMut(&dyn ServiceResolver) -> DDIResult<Box<dyn Any + Send>> + 'static + Send>),
    Fn(Box<dyn Fn(&dyn ServiceResolver) -> DDIResult<Box<dyn Any + Send>> + 'static + Send>),
}

#[derive(Debug, Clone, Copy)]
pub enum ServiceLifetime {
    Singleton,
    Transient,
    Scoped(u8),
}

impl Default for ServiceLifetime {
    fn default() -> Self {
        Self::Singleton
    }
}

struct ServiceFactoryDescriptor(ServiceLifetime, ServiceFactory);

pub struct ServiceCollection {
    pub map: HashMap<ServiceSymbol, Vec<(ServiceName, ServiceFactoryDescriptor)>>,
}

impl ServiceCollection {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn provider(self) -> ServiceProvider {
        ServiceProvider {
            collection: RefCell::new(self),
            cache: ServiceProviderCachePool::new(),
        }
    }
}

impl ServiceCollection {
    fn service_raw(
        &mut self,
        symbol: ServiceSymbol,
        name: ServiceName,
        factoryDescriptor: ServiceFactoryDescriptor,
    ) {
        if let Some(service) = self.map.get_mut(&symbol) {
            let exists = service.iter().position(|s| &s.0 == &name);
            if let Some(exists) = exists {
                service[exists] = (name, factoryDescriptor)
            } else {
                service.push((name, factoryDescriptor))
            }
        } else {
            self.map.insert(symbol, vec![(name, factoryDescriptor)]);
        }
    }
}

impl Debug for ServiceCollection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        for (symbol, s) in self.map.iter() {
            for (n, _) in s.iter() {
                list.entry(&service_symbol_debug_name(symbol, n));
            }
        }
        list.finish()
    }
}

pub trait ServiceCollectionExt: Sized {
    fn service<T: 'static + Send>(&mut self, value: T);

    fn service_var<T: 'static + Send>(&mut self, name: impl Into<ServiceName>, value: T);

    fn service_factory<
        Param,
        T: 'static + Send,
        Factory: ServiceFnOnce<Param, DDIResult<T>> + 'static + Send + Sync,
    >(
        &mut self,
        factory: Factory,
    );

    fn service_factory_var<
        Param,
        T: 'static + Send,
        Factory: ServiceFnOnce<Param, DDIResult<T>> + 'static + Send + Sync,
    >(
        &mut self,
        name: impl Into<ServiceName>,
        factory: Factory,
    );
}

impl ServiceCollectionExt for ServiceCollection {
    fn service<T: 'static + Send>(&mut self, value: T) {
        self.service_var("default", value)
    }

    fn service_var<T: 'static + Send>(&mut self, name: impl Into<ServiceName>, value: T) {
        let symbol = ServiceSymbol::new::<T>();
        self.service_raw(
            symbol,
            name.into(),
            ServiceFactoryDescriptor(
                ServiceLifetime::Singleton,
                ServiceFactory::FnOnce(Some(Box::new(move |_| Ok(Box::new(value))))),
            ),
        );
    }

    fn service_factory<
        Param,
        T: 'static + Send,
        Factory: ServiceFnOnce<Param, DDIResult<T>> + 'static + Send + Sync,
    >(
        &mut self,
        factory: Factory,
    ) {
        self.service_factory_var("default", factory)
    }

    fn service_factory_var<
        Param,
        T: 'static + Send,
        Factory: ServiceFnOnce<Param, DDIResult<T>> + 'static + Send + Sync,
    >(
        &mut self,
        name: impl Into<ServiceName>,
        factory: Factory,
    ) {
        let symbol = ServiceSymbol::new::<T>();
        self.service_raw(
            symbol,
            name.into(),
            ServiceFactoryDescriptor(
                ServiceLifetime::Singleton,
                ServiceFactory::FnOnce(Some(Box::new(move |service_resolver| {
                    Ok(Box::new(factory.run_with_once(service_resolver)??))
                }))),
            ),
        )
    }
}

struct ServiceProviderCachePool {
    // SAFETY: UnsafeCell is designed to be able to write new data while holding read references to the existing data in the hashmap. We only need to ensure that writing to the hashmap does not affect the existing data, i.e., we can safely hold read references to the data in the hashmap.
    cache: UnsafeCell<HashMap<(ServiceSymbol, ServiceName), ServiceProviderCacheItem>>,
}

struct ServiceProviderCacheItem {
    owned: Box<dyn Any + Send>,
}

impl ServiceProviderCacheItem {
    fn as_any_ref(&self) -> &(dyn Any + Send) {
        self.owned.as_ref()
    }
}

impl ServiceProviderCachePool {
    fn new() -> Self {
        Self {
            cache: Default::default(),
        }
    }

    fn get_cache_ref(
        &self,
        symbol: ServiceSymbol,
        var_name: ServiceName,
    ) -> Option<&(dyn Any + Send)> {
        // SAFETY: See the comments for the cache field
        unsafe { &*self.cache.get() }
            .get(&(symbol, var_name))
            .map(|c| c.as_any_ref())
    }

    fn cache_insert_owned(
        &self,
        symbol: ServiceSymbol,
        var_name: ServiceName,
        value: Box<dyn Any + Send>,
    ) -> &(dyn Any + Send) {
        // SAFETY: See the comments for the cache field
        let cache = unsafe { &mut *self.cache.get() };

        match cache.entry((symbol.clone(), var_name.clone())) {
            std::collections::hash_map::Entry::Occupied(_) => panic!("cache state error!"),
            std::collections::hash_map::Entry::Vacant(entry) => entry
                .insert(ServiceProviderCacheItem { owned: value })
                .as_any_ref(),
        }
    }
}

pub trait ServiceResolver {
    fn resolve(&self, symbol: ServiceSymbol, var_name: ServiceName)
        -> DDIResult<&(dyn Any + Send)>;

    fn resolve_all(
        &self,
        symbol: ServiceSymbol,
    ) -> DDIResult<Vec<(ServiceName, &(dyn Any + Send))>>;
}

impl<T: ServiceResolver + ?Sized> ServiceResolver for &T {
    fn resolve(
        &self,
        symbol: ServiceSymbol,
        var_name: ServiceName,
    ) -> DDIResult<&(dyn Any + Send)> {
        (*self).resolve(symbol, var_name)
    }

    fn resolve_all(
        &self,
        symbol: ServiceSymbol,
    ) -> DDIResult<Vec<(ServiceName, &(dyn Any + Send))>> {
        (*self).resolve_all(symbol)
    }
}

pub trait ServiceResolverExt: ServiceResolver {
    fn get<T: 'static>(&self) -> DDIResult<&T> {
        let symbol = ServiceSymbol::new::<T>();
        Ok(self
            .resolve(symbol, "default".into())?
            .downcast_ref()
            .unwrap())
    }

    fn get_var<T: 'static>(&self, name: impl Into<ServiceName>) -> DDIResult<&T> {
        let symbol = ServiceSymbol::new::<T>();
        Ok(self.resolve(symbol, name.into())?.downcast_ref().unwrap())
    }

    fn get_all<T: 'static>(&self) -> DDIResult<Vec<(ServiceName, &T)>> {
        let symbol = ServiceSymbol::new::<T>();
        Ok(self
            .resolve_all(symbol)?
            .into_iter()
            .map(|(name, v)| (name, v.downcast_ref().unwrap()))
            .collect())
    }

    fn wrap<'p>(&'p self, services: ServiceProvider) -> ChildServiceProvider<'p>
    where
        &'p Self: ServiceResolver,
    {
        ChildServiceProvider::<'p> {
            parent: Box::new(self) as Box<dyn ServiceResolver + 'p>,
            this: services,
        }
    }
}

impl<Resolver> ServiceResolverExt for Resolver where Resolver: ServiceResolver {}

struct TrackServiceResolver<'r> {
    provider: &'r ServiceProvider,
    depth: usize,
    #[cfg(debug_assertions)]
    track: Vec<(ServiceSymbol, ServiceName)>,
}

impl<'r> TrackServiceResolver<'r> {
    fn new(provider: &'r ServiceProvider) -> TrackServiceResolver<'r> {
        Self {
            provider,
            depth: 0,
            #[cfg(debug_assertions)]
            track: Vec::new(),
        }
    }

    fn resolve_inner(
        &self,
        symbol: ServiceSymbol,
        var_name: ServiceName,
    ) -> DDIResult<&'r (dyn Any + Send)> {
        if let Some(value) = self.provider.cache.get_cache_ref(symbol, var_name.clone()) {
            return Ok(value);
        }

        let mut collection = self.provider.collection.borrow_mut();
        let vars = collection.map.get_mut(&symbol);
        let factory = vars
            .and_then(|vars| {
                vars.iter_mut()
                    .find(|s| &s.0 == &var_name)
                    .map(|s| &mut s.1)
            })
            .ok_or_else(|| DDIError::ServiceNotFound((symbol, var_name.clone())))?;

        let factory = factory.take();
        core::mem::drop(collection);
        let mut next_resolver = self.track(&symbol, &var_name)?;

        let value = {
            let service = factory.unwrap()(&mut next_resolver).map_err(|e| {
                if let DDIError::ServiceNotFound((not_found_symbol, not_found_name)) = e {
                    DDIError::MissingDependency(
                        (symbol, var_name.clone()),
                        (not_found_symbol, not_found_name),
                    )
                } else {
                    e
                }
            })?;

            self.provider
                .cache
                .cache_insert_owned(symbol, var_name, service)
        };
        Ok(value)
    }

    fn resolve_all_inner(
        &self,
        symbol: ServiceSymbol,
    ) -> DDIResult<Vec<(ServiceName, &'r (dyn Any + Send))>> {
        let mut collection = self.provider.collection.borrow_mut();
        let vars = collection.map.get_mut(&symbol).map(|v| &mut v[..]);

        if let Some(vars) = vars {
            let mut result = Vec::with_capacity(vars.len());
            let mut factories = Vec::with_capacity(vars.len());

            for (var_name, factory) in vars.iter_mut() {
                if let Some(value) = self
                    .provider
                    .cache
                    .get_cache_ref(symbol.clone(), var_name.clone())
                {
                    result.push((var_name.clone(), value));
                } else {
                    factories.push((var_name.clone(), factory.take()));
                }
            }

            core::mem::drop(collection);

            for (var_name, factory) in factories {
                let mut next_resolver = self.track(&symbol, &var_name)?;
                let service = factory.unwrap()(&mut next_resolver).map_err(|e| {
                    if let DDIError::ServiceNotFound((not_found_symbol, not_found_name)) = e {
                        DDIError::MissingDependency(
                            (symbol.clone(), var_name.clone()),
                            (not_found_symbol, not_found_name),
                        )
                    } else {
                        e
                    }
                })?;

                let value = self.provider.cache.cache_insert_owned(
                    symbol.clone(),
                    var_name.clone(),
                    service,
                );
                result.push((var_name, value));
            }
            Ok(result)
        } else {
            Ok(Vec::new())
        }
    }

    fn track<'nr>(
        &'nr self,
        symbol: &ServiceSymbol,
        var_name: &ServiceName,
    ) -> DDIResult<TrackServiceResolver<'nr>> {
        let depth = self.depth + 1;
        if depth >= 1000 {
            return Err(DDIError::RecursionLimit);
        }

        #[cfg(debug_assertions)]
        let track = {
            let circular = self
                .track
                .iter()
                .find(|(s, v)| s == symbol && v == var_name)
                .is_some();
            let mut track = self.track.clone();
            track.push((symbol.clone(), var_name.clone()));
            if circular {
                return Err(DDIError::CircularDependencyDetected(track));
            }
            track
        };

        Ok(TrackServiceResolver::<'nr> {
            provider: self.provider,
            depth,
            #[cfg(debug_assertions)]
            track,
        })
    }
}

impl<'r> ServiceResolver for TrackServiceResolver<'r> {
    fn resolve(
        &self,
        symbol: ServiceSymbol,
        var_name: ServiceName,
    ) -> DDIResult<&(dyn Any + Send)> {
        self.resolve_inner(symbol, var_name)
    }

    fn resolve_all(
        &self,
        symbol: ServiceSymbol,
    ) -> DDIResult<Vec<(ServiceName, &(dyn Any + Send))>> {
        self.resolve_all_inner(symbol)
    }
}

pub struct ServiceRef {
    resolver: &'static dyn ServiceResolver,
}

impl ServiceRef {
    pub fn resolver(&self) -> &dyn ServiceResolver {
        self.resolver
    }
}

impl ServiceResolver for ServiceRef {
    fn resolve(
        &self,
        symbol: ServiceSymbol,
        var_name: ServiceName,
    ) -> DDIResult<&(dyn Any + Send)> {
        self.resolver.resolve(symbol, var_name)
    }

    fn resolve_all(
        &self,
        symbol: ServiceSymbol,
    ) -> DDIResult<Vec<(ServiceName, &(dyn Any + Send))>> {
        self.resolver.resolve_all(symbol)
    }
}

pub struct ServiceProvider {
    collection: RefCell<ServiceCollection>,
    cache: ServiceProviderCachePool,
}

impl ServiceResolver for ServiceProvider {
    fn resolve(
        &self,
        symbol: ServiceSymbol,
        var_name: ServiceName,
    ) -> DDIResult<&(dyn Any + Send)> {
        TrackServiceResolver::new(self).resolve_inner(symbol, var_name)
    }

    fn resolve_all(
        &self,
        symbol: ServiceSymbol,
    ) -> DDIResult<Vec<(ServiceName, &(dyn Any + Send))>> {
        TrackServiceResolver::new(self).resolve_all_inner(symbol)
    }
}

pub struct ChildServiceProvider<'p> {
    parent: Box<dyn ServiceResolver + 'p>,
    this: ServiceProvider,
}

impl ServiceResolver for ChildServiceProvider<'_> {
    fn resolve(
        &self,
        symbol: ServiceSymbol,
        var_name: ServiceName,
    ) -> DDIResult<&(dyn Any + Send)> {
        match self.this.resolve(symbol, var_name.clone()) {
            Ok(service) => Ok(service),
            Err(DDIError::ServiceNotFound(_)) => self.parent.resolve(symbol, var_name),
            Err(err) => Err(err),
        }
    }

    fn resolve_all(
        &self,
        symbol: ServiceSymbol,
    ) -> DDIResult<Vec<(ServiceName, &(dyn Any + Send))>> {
        match self.this.resolve_all(symbol.clone()) {
            Ok(services) => match self.parent.resolve_all(symbol) {
                Ok(parent_services) => {
                    let mut result = Vec::with_capacity(services.len() + parent_services.len());
                    result.extend(services);
                    result.extend(parent_services);
                    result.dedup_by_key(|i| i.0.clone());
                    Ok(result)
                }
                Err(DDIError::ServiceNotFound(_)) => Ok(services),
                Err(err) => Err(err),
            },
            Err(DDIError::ServiceNotFound(_)) => self.parent.resolve_all(symbol),
            Err(err) => Err(err),
        }
    }
}

pub enum DDIError {
    RecursionLimit,
    CircularDependencyDetected(Vec<(ServiceSymbol, ServiceName)>),
    MissingDependency((ServiceSymbol, ServiceName), (ServiceSymbol, ServiceName)),
    ServiceNotFound((ServiceSymbol, ServiceName)),
}

impl Debug for DDIError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DDIError::RecursionLimit => {
                write!(
                    fmt,
                    "Trigger recursion limit, most likely with circular dependencies."
                )
            }
            DDIError::CircularDependencyDetected(list) => {
                write!(fmt, "A circular dependency was detected.\n")?;
                for (i, (symbol, var_name)) in list.iter().enumerate() {
                    write!(fmt, "{}", service_symbol_debug_name(symbol, var_name))?;
                    if i < list.len() - 1 {
                        // not the end
                        write!(fmt, " -> ")?;
                    }
                }
                Ok(())
            }
            DDIError::MissingDependency(
                (source, source_var_name),
                (dependency, dependency_var_name),
            ) => {
                write!(
                    fmt,
                    "Missing dependency {} in creating service {}.",
                    service_symbol_debug_name(dependency, dependency_var_name),
                    service_symbol_debug_name(source, source_var_name),
                )
            }
            DDIError::ServiceNotFound((symbol, symbol_var_name)) => {
                write!(
                    fmt,
                    "Service not found {}.",
                    service_symbol_debug_name(symbol, symbol_var_name)
                )
            }
        }
    }
}

impl Display for DDIError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl error::Error for DDIError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}

pub type DDIResult<T> = std::result::Result<T, DDIError>;

fn service_symbol_debug_name(symbol: &ServiceSymbol, var_name: &ServiceName) -> String {
    if var_name.name() == "default" {
        format!("[{:?}]", symbol)
    } else {
        format!("[{:?}]({})", symbol, var_name)
    }
}

#[cfg(test)]
mod tests {

    use crate::{DDIError, ServiceCollectionExt, ServiceFn, ServiceResolverExt, ServiceSymbol};

    use super::ServiceCollection;

    #[derive(Debug)]
    struct TestService(String);

    #[test]
    fn service() {
        let mut services = ServiceCollection::new();
        services.service(1usize);
        services.service("helloworld");
        services
            .service_factory(|num: &usize, str: &&str| Ok(TestService(format!("{}{}", num, str))));
        assert_eq!(services.len(), 3);

        let provider = services.provider();
        assert_eq!(provider.get::<TestService>().unwrap().0, "1helloworld");
    }

    #[test]
    fn service_not_found() {
        let mut services = ServiceCollection::new();
        services.service(1usize);
        services.service("helloworld");

        let provider = services.provider();
        assert!(matches!(
            provider.get::<TestService>().unwrap_err(),
            DDIError::ServiceNotFound((s,n)) if s == ServiceSymbol::new::<TestService>() && n == "default".into()
        ));
    }

    #[test]
    fn service_missing_dependency() {
        let mut services = ServiceCollection::new();
        services.service("helloworld");
        services
            .service_factory(|num: &usize, str: &&str| Ok(TestService(format!("{}{}", num, str))));

        let provider = services.provider();
        assert!(matches!(
            provider.get::<TestService>().unwrap_err(),
            DDIError::MissingDependency((s,n), (ds, dn)) if s == ServiceSymbol::new::<TestService>() && n == "default".into() && ds == ServiceSymbol::new::<usize>() && dn == "default".into()
        ));
    }

    #[test]
    fn service_circular_dependency() {
        let mut services = ServiceCollection::new();
        services.service(1usize);
        services.service("helloworld");
        services.service_factory(|num: &usize, str: &&str, _: &TestService| {
            Ok(TestService(format!("{}{}", num, str)))
        });

        let provider = services.provider();
        assert!(matches!(
            provider.get::<TestService>().unwrap_err(),
            DDIError::CircularDependencyDetected(v) if v.as_slice() == &[(ServiceSymbol::new::<TestService>(), "default".into()),(ServiceSymbol::new::<TestService>(), "default".into())]
        ));

        let mut services = ServiceCollection::new();
        services.service(1usize);
        services.service_factory(|_: &TestService| Ok("helloworld"));
        services.service_factory(|num: &usize, str: &&str, _: &TestService| {
            Ok(TestService(format!("{}{}", num, str)))
        });

        let provider = services.provider();
        assert!(matches!(
            provider.get::<TestService>().unwrap_err(),
            DDIError::CircularDependencyDetected(v) if v.as_slice() == &[(ServiceSymbol::new::<TestService>(), "default".into()), (ServiceSymbol::new::<&str>(), "default".into()),(ServiceSymbol::new::<TestService>(), "default".into())]
        ));
    }

    #[test]
    fn service_function() {
        let mut services = ServiceCollection::new();
        services.service(1usize);
        services.service("helloworld");

        assert_eq!(
            (|num: &usize, str: &&str| format!("{}{}", num, str))
                .run_with(&mut services.provider())
                .unwrap(),
            "1helloworld".to_owned()
        )
    }

    #[test]
    fn service_wrap() {
        let mut services = ServiceCollection::new();
        services.service(1usize);
        let provider = services.provider();

        let mut child_services = ServiceCollection::new();
        child_services.service(2usize);

        let wrapped = provider.wrap(child_services.provider());
        assert_eq!(&2usize, wrapped.get::<usize>().unwrap());
    }
}
