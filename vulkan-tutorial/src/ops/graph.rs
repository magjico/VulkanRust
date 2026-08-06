//===============================================
// Graph (Tree like and Index-Based Design)
//===============================================
use std::slice::{Iter, IterMut};

use crate::type_safety::NodeId;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Node<T> {
	pub parent: Option<NodeId>,
	pub childs: Vec<NodeId>,

	pub value: T
}

#[derive(Debug, Clone)]
pub struct FlatGraph<T>(Vec<Node<T>>);

impl<T> Default for FlatGraph<T> {
	fn default() -> Self {
		FlatGraph {0: vec![]}
	}
}

impl<T> FlatGraph<T> {
	pub fn new(nodes: Vec<Node<T>>) -> Self {
		FlatGraph {0: nodes}
	}

	#[inline]
	pub fn get(&self, id: NodeId) -> &Node<T> {
		self.0.get(id.0)
			.unwrap_or_else(|| panic!("graph index {} out of bounds for length {}", id.0, self.0.len()))
	}

	#[inline]
	pub fn get_mut(&mut self, id: NodeId) -> &mut Node<T> {
		let len = self.0.len();
		self.0.get_mut(id.0)
			.unwrap_or_else(|| panic!("graph index {} out of bounds for length {}", id.0, len))
	}

	pub fn add_node(&mut self, value: T, parent: Option<NodeId>) -> NodeId {
		let id = NodeId(self.0.len());
		self.0.push(Node {parent, childs: Vec::new(), value});
		if let Some(p) = parent {
			self.0[p.0].childs.push(id);
		}
		
		id
	}

	#[inline]
	pub fn iter(&self) -> Iter<'_, Node<T>> {
		self.0.iter()
	}
	#[inline]
	pub fn iter_mut(&mut self) -> IterMut<'_, Node<T>> {
		self.0.iter_mut()
	}
}