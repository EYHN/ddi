#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::{boxed::Box, format, string::String, vec, vec::Vec};
use core::{
    any::{type_name, TypeId},
    fmt::{self, Debug, Display},
    hash::Hash,
};

#[cfg(not(feature = "sync"))]
use core::cell::{RefCell, UnsafeCell};

#[cfg(feature = "std")]
use std::error;

#[cfg(feature = "sync")]
use std::sync::{Mutex, RwLock, TryLockError};

#[cfg(not(feature = "std"))]
pub type Map<K, V> = alloc::collections::BTreeMap<K, V>;
#[cfg(not(feature = "std"))]
pub type MapEntry<'a, K, V> = alloc::collections::btree_map::Entry<'a, K, V>;

#[cfg(feature = "std")]
pub type Map<K, V> = std::collections::HashMap<K, V>;
#[cfg(feature = "std")]
pub type MapEntry<'a, K, V> = std::collections::hash_map::Entry<'a, K, V>;

#[cfg(not(feature = "sync"))]
mod cfg {
    use core::any::Any;

    pub type AnyService = dyn Any;
    pub trait TypedServiceTrait: 'static {}
    impl<T> TypedServiceTrait for T where T: 'static {}
    pub type Rc<T> = alloc::rc::Rc<T>;
    pub type Weak<T> = alloc::rc::Weak<T>;
}

#[cfg(feature = "sync")]
mod cfg {
    use core::any::Any;

    pub type AnyService = dyn Any + Send + Sync;
    pub trait TypedServiceTrait: 'static + Send + Sync {}
    impl<T> TypedServiceTrait for T where T: 'static + Send + Sync {}
    pub type Rc<T> = alloc::sync::Arc<T>;
    pub type Weak<T> = alloc::sync::Weak<T>;
}

use cfg::*;

pub type Service<T> = Rc<T>;

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
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
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

impl PartialOrd for ServiceSymbol {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        match (self, other) {
            (Self::Type(type_id, _), Self::Type(type_id_other, _)) => {
                type_id.partial_cmp(type_id_other)
            }
        }
    }
}

impl Ord for ServiceSymbol {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        match (self, other) {
            (Self::Type(type_id, _), Self::Type(type_id_other, _)) => type_id.cmp(type_id_other),
        }
    }
}

#[derive(Clone, PartialOrd, Ord)]
pub enum ServiceName {
    Static(&'static str),
    Dynamic(Rc<String>),
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
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
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
        core::fmt::Debug::fmt(self.name(), f)
    }
}

impl Display for ServiceName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        core::fmt::Display::fmt(self.name(), f)
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
        ServiceName::Dynamic(Rc::new(s))
    }
}

pub trait ServiceFnOnce<Param, Out> {
    fn run_once(self, dependencies: &[&AnyService]) -> Out;
    fn dependencies() -> Vec<(ServiceSymbol, ServiceName)>;
    fn run_with_once(self, service_ref: &dyn ServiceResolver) -> DDIResult<Out>;
}

pub trait ServiceFnMut<Param, Out>: ServiceFnOnce<Param, Out> {
    fn run_mut(&mut self, dependencies: &[&AnyService]) -> Out;
    fn run_with_mut(&mut self, service_ref: &dyn ServiceResolver) -> DDIResult<Out>;
}

pub trait ServiceFn<Param, Out>: ServiceFnMut<Param, Out> {
    fn run(&self, dependencies: &[&AnyService]) -> Out;
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
            fn run(&self, dependencies: &[&AnyService]) -> Out {
                fn call_inner<Out, $($param,)*>(f: impl Fn($(&$param,)*) -> Out, $($param: &$param,)*) -> Out {
                    f($($param,)*)
                }
                if let [$($param,)*] = dependencies {
                    $(let $param = $param.downcast_ref::<$param>().expect(&format!("Failed downcast to {}", core::any::type_name::<$param>()));)*
                    call_inner(self, $($param,)*)
                } else {
                    unreachable!()
                }
            }

            fn run_with(&self, service: &dyn ServiceResolver) -> DDIResult<Out> {
                let dependencies = Self::dependencies();
                let mut deps = Vec::with_capacity(dependencies.len());
                for (dep_symbol, dep_name) in dependencies.into_iter() {
                    let value = service.resolve(dep_symbol.clone(), dep_name)?;
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
            fn run_mut(&mut self, dependencies: &[&AnyService]) -> Out {
                fn call_inner<Out, $($param,)*>(mut f: impl FnMut($(&$param,)*) -> Out, $($param: &$param,)*) -> Out {
                    f($($param,)*)
                }
                if let [$($param,)*] = dependencies {
                    $(let $param = $param.downcast_ref::<$param>().expect(&format!("Failed downcast to {}", core::any::type_name::<$param>()));)*
                    call_inner(self, $($param,)*)
                } else {
                    unreachable!()
                }
            }

            fn run_with_mut(&mut self, service: &dyn ServiceResolver) -> DDIResult<Out> {
                let dependencies = Self::dependencies();
                let mut deps = Vec::with_capacity(dependencies.len());
                for (dep_symbol, dep_name) in dependencies.into_iter() {
                    let value = service.resolve(dep_symbol.clone(), dep_name)?;
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
            fn run_once(self, dependencies: &[&AnyService]) -> Out {
                fn call_inner<Out, $($param,)*>(f: impl FnOnce($(&$param,)*) -> Out, $($param: &$param,)*) -> Out {
                    f($($param,)*)
                }
                if let [$($param,)*] = dependencies {
                    $(let $param = $param.downcast_ref::<$param>().expect(&format!("Failed downcast to {}", core::any::type_name::<$param>()));)*
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
                let mut deps = Vec::with_capacity(dependencies.len());
                for (dep_symbol, dep_name) in dependencies.into_iter() {
                    let value = service.resolve(dep_symbol, dep_name)?;
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

#[cfg(not(feature = "sync"))]
struct ServiceFactory(
    RefCell<Option<Box<dyn FnOnce(&dyn ServiceResolver) -> DDIResult<Box<AnyService>> + 'static>>>,
);

#[cfg(not(feature = "sync"))]
impl ServiceFactory {
    fn new(f: impl FnOnce(&dyn ServiceResolver) -> DDIResult<Box<AnyService>> + 'static) -> Self {
        Self(RefCell::new(Some(Box::new(f))))
    }
    fn take(
        &self,
    ) -> DDIResult<Box<dyn FnOnce(&dyn ServiceResolver) -> DDIResult<Box<AnyService>> + 'static>>
    {
        let lock = self.0.try_borrow_mut();
        match lock {
            Ok(mut factory) => Ok(factory.take().unwrap()),
            Err(_) => Err(DDIError::Deadlock),
        }
    }
}

#[cfg(feature = "sync")]
struct ServiceFactory(
    Mutex<
        Option<
            Box<dyn FnOnce(&dyn ServiceResolver) -> DDIResult<Box<AnyService>> + 'static + Send>,
        >,
    >,
);

#[cfg(feature = "sync")]
impl ServiceFactory {
    fn new(
        f: impl FnOnce(&dyn ServiceResolver) -> DDIResult<Box<AnyService>> + 'static + Send,
    ) -> Self {
        Self(Mutex::new(Some(Box::new(f))))
    }
    fn take(
        &self,
    ) -> DDIResult<
        Box<dyn FnOnce(&dyn ServiceResolver) -> DDIResult<Box<AnyService>> + 'static + Send>,
    > {
        let lock = self.0.try_lock();
        match lock {
            Ok(mut factory) => Ok(factory.take().unwrap()),
            Err(TryLockError::Poisoned(_)) => Err(DDIError::PoisonError),
            Err(TryLockError::WouldBlock) => Err(DDIError::Deadlock),
        }
    }
}

pub struct ServiceCollection {
    map: Map<ServiceSymbol, Vec<(ServiceName, ServiceFactory)>>,
}

impl ServiceCollection {
    pub fn new() -> Self {
        Self { map: Map::new() }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn provider(self) -> ServiceProvider {
        ServiceProvider::new(self, ServiceProviderCachePool::new())
    }

    fn service_raw(&mut self, symbol: ServiceSymbol, name: ServiceName, factory: ServiceFactory) {
        if let Some(service) = self.map.get_mut(&symbol) {
            let exists = service.iter().position(|s| &s.0 == &name);
            if let Some(exists) = exists {
                service[exists] = (name, factory)
            } else {
                service.push((name, factory))
            }
        } else {
            self.map.insert(symbol, vec![(name, factory)]);
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
    fn service<T: TypedServiceTrait>(&mut self, value: T);

    fn service_var<T: TypedServiceTrait>(&mut self, name: impl Into<ServiceName>, value: T);

    fn service_factory<
        Param,
        T: TypedServiceTrait,
        Factory: ServiceFnOnce<Param, DDIResult<T>> + TypedServiceTrait,
    >(
        &mut self,
        factory: Factory,
    );

    fn service_factory_var<
        Param,
        T: TypedServiceTrait,
        Factory: ServiceFnOnce<Param, DDIResult<T>> + TypedServiceTrait,
    >(
        &mut self,
        name: impl Into<ServiceName>,
        factory: Factory,
    );
}

impl ServiceCollectionExt for ServiceCollection {
    fn service<T: TypedServiceTrait>(&mut self, value: T) {
        self.service_var("default", value)
    }

    fn service_var<T: TypedServiceTrait>(&mut self, name: impl Into<ServiceName>, value: T) {
        let symbol = ServiceSymbol::new::<T>();
        self.service_raw(
            symbol,
            name.into(),
            ServiceFactory::new(move |_| Ok(Box::new(value))),
        );
    }

    fn service_factory<
        Param,
        T: TypedServiceTrait,
        Factory: ServiceFnOnce<Param, DDIResult<T>> + TypedServiceTrait,
    >(
        &mut self,
        factory: Factory,
    ) {
        self.service_factory_var("default", factory)
    }

    fn service_factory_var<
        Param,
        T: TypedServiceTrait,
        Factory: ServiceFnOnce<Param, DDIResult<T>> + TypedServiceTrait,
    >(
        &mut self,
        name: impl Into<ServiceName>,
        factory: Factory,
    ) {
        let symbol = ServiceSymbol::new::<T>();
        self.service_raw(
            symbol,
            name.into(),
            ServiceFactory::new(move |service_resolver| {
                Ok(Box::new(factory.run_with_once(service_resolver)??))
            }),
        )
    }
}

#[cfg(not(feature = "sync"))]
struct InsertOnlyMap<K, V> {
    map: UnsafeCell<Map<K, Box<V>>>,
}

#[cfg(not(feature = "sync"))]
impl<K, V> Default for InsertOnlyMap<K, V> {
    fn default() -> Self {
        Self {
            map: Default::default(),
        }
    }
}

#[cfg(not(feature = "sync"))]
impl<K: Clone + Ord + Hash, V> InsertOnlyMap<K, V> {
    fn insert_with(&self, k: K, f: impl FnOnce() -> DDIResult<V>) -> DDIResult<&V> {
        Ok(unsafe {
            let map = self.map.get();
            match (*map).entry(k) {
                MapEntry::Occupied(entry) => {
                    &*((&**entry.get()) as *const _)
                }
                MapEntry::Vacant(entry) => {
                    &*((&**entry.insert(Box::new(f()?))) as *const _)
                }
            }
        })
    }
}

#[cfg(feature = "sync")]
struct InsertOnlyMap<K, V> {
    map: RwLock<Map<K, Box<V>>>,
}

#[cfg(feature = "sync")]
impl<K, V> Default for InsertOnlyMap<K, V> {
    fn default() -> Self {
        Self {
            map: Default::default(),
        }
    }
}

#[cfg(feature = "sync")]
impl<K: Eq + Hash, V> InsertOnlyMap<K, V> {
    fn insert_with(&self, k: K, f: impl FnOnce() -> DDIResult<V>) -> DDIResult<&V> {
        match self.map.write() {
            Ok(mut map) => {
                let ret = unsafe {
                    match map.entry(k) {
                        std::collections::hash_map::Entry::Occupied(entry) => {
                            &*((&**entry.get()) as *const _)
                        }
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            &*((&**entry.insert(Box::new(f()?))) as *const _)
                        }
                    }
                };
                Ok(ret)
            }
            Err(_) => Err(DDIError::PoisonError),
        }
    }
}

struct ServiceProviderCachePool {
    cache: InsertOnlyMap<ServiceSymbol, InsertOnlyMap<ServiceName, Box<AnyService>>>,
}

impl ServiceProviderCachePool {
    fn new() -> Self {
        Self {
            cache: Default::default(),
        }
    }

    fn get_or_insert(
        &self,
        symbol: ServiceSymbol,
        var_name: ServiceName,
        insert: impl FnOnce() -> DDIResult<Box<AnyService>>,
    ) -> DDIResult<&AnyService> {
        let part = self.cache.insert_with(symbol, || Ok(Default::default()))?;
        let service = part.insert_with(var_name, move || insert())?;
        Ok(service.as_ref())
    }
}

pub trait ServiceResolver {
    fn resolve(&self, symbol: ServiceSymbol, var_name: ServiceName) -> DDIResult<&AnyService>;

    fn resolve_all(&self, symbol: ServiceSymbol) -> DDIResult<Vec<(ServiceName, &AnyService)>>;
}

impl<T: ServiceResolver + ?Sized> ServiceResolver for &T {
    fn resolve(&self, symbol: ServiceSymbol, var_name: ServiceName) -> DDIResult<&AnyService> {
        (*self).resolve(symbol, var_name)
    }

    fn resolve_all(&self, symbol: ServiceSymbol) -> DDIResult<Vec<(ServiceName, &AnyService)>> {
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
    ) -> DDIResult<&'r AnyService> {
        let cache = &self.provider.shared.cache;
        let next_resolver = self.track(&symbol, &var_name)?;
        cache.get_or_insert(symbol, var_name.clone(), move || {
            let collection = &self.provider.shared.collection;
            let vars = collection.map.get(&symbol);
            let factory = vars
                .and_then(|vars| vars.iter().find(|s| &s.0 == &var_name).map(|s| &s.1))
                .ok_or_else(|| DDIError::ServiceNotFound((symbol, var_name.clone())))?;

            let factory = factory.take()?;
            core::mem::drop(collection);

            let service = factory(&next_resolver).map_err(|e| {
                if let DDIError::ServiceNotFound((not_found_symbol, not_found_name)) = e {
                    DDIError::MissingDependency(
                        (symbol, var_name.clone()),
                        (not_found_symbol, not_found_name),
                    )
                } else {
                    e
                }
            })?;

            Ok(service)
        })
    }

    fn resolve_all_inner(
        &self,
        symbol: ServiceSymbol,
    ) -> DDIResult<Vec<(ServiceName, &'r AnyService)>> {
        let collection = &self.provider.shared.collection;
        let vars = collection.map.get(&symbol).map(|v| &v[..]);

        if let Some(vars) = vars {
            let cache = &self.provider.shared.cache;
            let mut result = Vec::with_capacity(vars.len());

            for (var_name, factory) in vars.iter() {
                let next_resolver = self.track(&symbol, &var_name)?;
                let service = cache.get_or_insert(symbol, var_name.clone(), move || {
                    let factory = factory.take()?;
                    let service = factory(&next_resolver).map_err(|e| {
                        if let DDIError::ServiceNotFound((not_found_symbol, not_found_name)) = e {
                            DDIError::MissingDependency(
                                (symbol, var_name.clone()),
                                (not_found_symbol, not_found_name),
                            )
                        } else {
                            e
                        }
                    })?;

                    Ok(service)
                })?;
                result.push((var_name.clone(), service))
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
            let circular = self.track.iter().find(|(s, _)| s == symbol).is_some();
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
    fn resolve(&self, symbol: ServiceSymbol, var_name: ServiceName) -> DDIResult<&AnyService> {
        self.resolve_inner(symbol, var_name)
    }

    fn resolve_all(&self, symbol: ServiceSymbol) -> DDIResult<Vec<(ServiceName, &AnyService)>> {
        self.resolve_all_inner(symbol)
    }
}

struct ServiceProviderInner {
    collection: ServiceCollection,
    cache: ServiceProviderCachePool,
}

impl ServiceProviderInner {
    fn new(collection: ServiceCollection, cache: ServiceProviderCachePool) -> Self {
        Self { collection, cache }
    }
}

#[derive(Clone)]
pub struct ServiceProvider {
    shared: Rc<ServiceProviderInner>,
}

impl ServiceProvider {
    fn new(collection: ServiceCollection, cache: ServiceProviderCachePool) -> Self {
        let shared = Rc::new(ServiceProviderInner::new(collection, cache));
        let weak = Rc::downgrade(&shared);
        unsafe {
            (shared.as_ref() as *const _ as *mut ServiceProviderInner)
                .as_mut()
                .unwrap()
        }
        .collection
        .service_factory(move || {
            Ok(ServiceProvider {
                shared: Weak::upgrade(&weak).unwrap(),
            })
        });
        ServiceProvider { shared }
    }
}

impl ServiceResolver for ServiceProvider {
    fn resolve(&self, symbol: ServiceSymbol, var_name: ServiceName) -> DDIResult<&AnyService> {
        TrackServiceResolver::new(self).resolve_inner(symbol, var_name)
    }

    fn resolve_all(&self, symbol: ServiceSymbol) -> DDIResult<Vec<(ServiceName, &AnyService)>> {
        TrackServiceResolver::new(self).resolve_all_inner(symbol)
    }
}

pub struct ChildServiceProvider<'p> {
    parent: Box<dyn ServiceResolver + 'p>,
    this: ServiceProvider,
}

impl ServiceResolver for ChildServiceProvider<'_> {
    fn resolve(&self, symbol: ServiceSymbol, var_name: ServiceName) -> DDIResult<&AnyService> {
        match self.this.resolve(symbol, var_name.clone()) {
            Ok(service) => Ok(service),
            Err(DDIError::ServiceNotFound(_)) => self.parent.resolve(symbol, var_name),
            Err(err) => Err(err),
        }
    }

    fn resolve_all(&self, symbol: ServiceSymbol) -> DDIResult<Vec<(ServiceName, &AnyService)>> {
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
    Deadlock,
    PoisonError,
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
            DDIError::Deadlock => {
                write!(fmt, "Deadlock.",)
            }
            DDIError::PoisonError => {
                write!(fmt, "Poison.",)
            }
        }
    }
}

impl Display for DDIError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[cfg(feature = "std")]
impl error::Error for DDIError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}

pub type DDIResult<T> = core::result::Result<T, DDIError>;

fn service_symbol_debug_name(symbol: &ServiceSymbol, var_name: &ServiceName) -> String {
    if var_name.name() == "default" {
        format!("[{:?}]", symbol)
    } else {
        format!("[{:?}]({})", symbol, var_name)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        DDIError, ServiceCollectionExt, ServiceFn, ServiceProvider, ServiceResolverExt,
        ServiceSymbol,
    };

    use alloc::{format, string::String};

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
            "1helloworld"
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

    struct ServiceOwnedProvider(ServiceProvider);

    impl ServiceOwnedProvider {
        fn get_string(&self) -> &str {
            *self.0.get::<&str>().unwrap()
        }
    }

    #[test]
    fn service_owned_provider() {
        let mut services = ServiceCollection::new();
        services.service("helloworld");
        services.service_factory(move |provider: &ServiceProvider| {
            Ok(ServiceOwnedProvider(provider.clone()))
        });

        let provider = services.provider();
        assert_eq!(
            provider.get::<ServiceOwnedProvider>().unwrap().get_string(),
            "helloworld"
        );
    }
}
