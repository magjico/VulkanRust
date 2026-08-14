//===============================================
// Graph (Tree like and Index-Based Design)
//===============================================
use std::slice::{Iter, IterMut};
use std::mem;

use crate::type_safety::NodeId;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Node<T> {
	pub parent: Option<NodeId>,
	pub childs: Vec<NodeId>,

	pub value: T,
}


/// Represent a graph where the Node are stored into a one dimensional array.
/// You can get information about the nodes themself, their ID (index) and their depths levels.
#[derive(Debug, Clone)]
pub struct FlatGraph<T> {
	nodes: Vec<Node<T>>,
	depths: Vec<u32>,
	levels: Vec<Vec<NodeId>>,

	max_depth: u32,
}

impl<T> Default for FlatGraph<T> {
	fn default() -> Self {
		FlatGraph {
			nodes: vec![],
			depths: vec![],
			levels: vec![],
			max_depth: 0
		}
	}
}

impl<T> FlatGraph<T> {
	pub fn new(nodes: Vec<Node<T>>) -> Self {
		let depths = Self::compute_depths(&nodes);
		let max_depth = depths.iter().copied().max().unwrap_or(0);
		let levels = Self::compute_depths_levels(&depths, &max_depth);
		FlatGraph {
			nodes,
			depths,
			max_depth,
			levels
		}
	}

	#[inline]
	pub fn get(&self, id: NodeId) -> &Node<T> {
		self.nodes.get(id.0)
			.unwrap_or_else(|| panic!("graph index {} out of bounds for length {}", id.0, self.nodes.len()))
	}

	#[inline]
	pub fn get_mut(&mut self, id: NodeId) -> &mut Node<T> {
		let len = self.nodes.len();
		self.nodes.get_mut(id.0)
			.unwrap_or_else(|| panic!("graph index {} out of bounds for length {}", id.0, len))
	}

	#[inline]
	pub fn get_depth(&self, id: NodeId) -> u32 {
		self.depths[id.0]
	}

	#[inline]
	pub fn get_max_depth(&self) -> u32 {
		self.max_depth
	}

	#[inline]
	pub fn get_level(&self, depth_level: u32) -> &[NodeId] {
		self.levels.get(depth_level as usize)
			.unwrap_or_else(|| panic!("graph depth level {} out of bounds for max depth {}", depth_level, self.max_depth))
	}

	pub fn add_node(&mut self, value: T, parent: Option<NodeId>) -> NodeId {
		let id = NodeId(self.nodes.len());
		self.nodes.push(Node {parent, childs: Vec::new(), value});
		if let Some(p) = parent {
			let depth = self.depths[p.0] + 1;
			self.nodes[p.0].childs.push(id);
			self.depths.push(depth);
			if depth > self.max_depth {
				self.max_depth = depth;
				self.levels.push(Vec::new());
			}
			self.levels[depth as usize].push(id);
		} else {
			self.depths.push(0);
			if self.levels.is_empty() { self.levels.push(Vec::new()); }
			self.levels[0].push(id);
		}
		
		id
	}

	#[inline]
	pub fn iter(&self) -> Iter<'_, Node<T>> {
		self.nodes.iter()
	}
	#[inline]
	pub fn iter_mut(&mut self) -> IterMut<'_, Node<T>> {
		self.nodes.iter_mut()
	}

	#[inline]
	pub fn len(&self) -> usize {
		self.nodes.len()
	}

	fn compute_depth(nodes: &[Node<T>], id: NodeId) -> u32 {
		let mut parent = nodes[id.0].parent;
		let mut depth = 0;

		while let Some(node_parent) = parent {
			depth += 1;
			parent = nodes[node_parent.0].parent;
		}

		depth
	}

	fn compute_depths(nodes: &[Node<T>]) -> Vec<u32> {
		(0..nodes.len()).map(|i| {
			Self::compute_depth(nodes, NodeId(i))
		}).collect()
	}

	fn compute_depths_levels(depths: &[u32], max_depth: &u32) -> Vec<Vec<NodeId>> {
		let mut levels = vec![Vec::new(); (max_depth + 1) as usize];

		for (index, &level) in depths.iter().enumerate() {
			levels[level as usize].push(NodeId(index));
		}

		levels
	}

	/// Use when you need bypass borrow-checker rules.
	/// 
	/// **warn**: if *func* panic then the [FlatGraph] levels will stay empty because of [std::mem::take].
	/// Normaly *func* never panics.
	pub fn with_levels<F, R>(&mut self, func: F) -> R 
		where F: FnOnce(&mut Self, &[Vec<NodeId>]) -> R
	{
		let levels = mem::take(&mut self.levels);
		let result = func(self, &levels);
		self.levels = levels;
		result
	}
}