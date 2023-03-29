/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file integer_set.cc
 * \brief The integer set functions
 */
#include <tvm/arith/int_set.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/arith/pattern.h>

#include <algorithm>
#include <unordered_map>
#include <utility>

#include "constraint_extract.h"
#include "integer_set.h"
#include "interval_set.h"

namespace tvm {
namespace arith {

using namespace tir;


void Update(const PrimExpr& constraint, const Array<Var>& vars, PresburgerSet& set) {
  auto constraints_union = ExtractComponents(constraint);
  auto space = set.getSpace();
  for (const PrimExpr& subconstraint : constraints_union) {
    auto entries = ExtractConstraints(subconstraint, false);
    IntegerPolyhedron subset(entries.size(), 0, vars.size() + 1, space);
    for (const PrimExpr& entry : entries) {
      if (entry.as<GENode>()) {
        auto coeffs_a = DetectLinearEquation(entry.as<GENode>()->a, vars);
        auto coeffs_b = DetectLinearEquation(entry.as<GENode>()->b, vars);
        std::vector<int64_t> int_coeffs;
        for (size_t i = 0; i < coeffs_a.size(); i++) {
          int_coeffs.push_back(*as_const_int(coeffs_a[i]) - *as_const_int(coeffs_b[i]));
        }
        subset.addInequality(int_coeffs);
      } else if (entry.as<GTNode>()) {
        auto coeffs_a = DetectLinearEquation(entry.as<GTNode>()->a, vars);
        auto coeffs_b = DetectLinearEquation(entry.as<GTNode>()->b, vars);
        std::vector<int64_t> int_coeffs;
        for (size_t i = 0; i < coeffs_a.size(); i++) {
          int_coeffs.push_back(*as_const_int(coeffs_a[i]) - *as_const_int(coeffs_b[i]));
        }
        int_coeffs[int_coeffs.size() - 1] -= 1;
        subset.addInequality(int_coeffs);
      } else if (entry.as<LENode>()) {
        auto coeffs_a = DetectLinearEquation(entry.as<LENode>()->a, vars);
        auto coeffs_b = DetectLinearEquation(entry.as<LENode>()->b, vars);
        std::vector<int64_t> int_coeffs;
        for (size_t i = 0; i < coeffs_a.size(); i++) {
          int_coeffs.push_back(*as_const_int(coeffs_b[i]) - *as_const_int(coeffs_a[i]));
        }
        subset.addInequality(int_coeffs);
      } else if (entry.as<LTNode>()) {
        auto coeffs_a = DetectLinearEquation(entry.as<LTNode>()->a, vars);
        auto coeffs_b = DetectLinearEquation(entry.as<LTNode>()->b, vars);
        std::vector<int64_t> int_coeffs;
        for (size_t i = 0; i < coeffs_a.size(); i++) {
          int_coeffs.push_back(*as_const_int(coeffs_b[i]) - *as_const_int(coeffs_a[i]));
        }
        int_coeffs[int_coeffs.size() - 1] -= 1;
        subset.addInequality(int_coeffs);
      } else if (entry.as<EQNode>()) {
        auto coeffs_a = DetectLinearEquation(entry.as<EQNode>()->a, vars);
        auto coeffs_b = DetectLinearEquation(entry.as<EQNode>()->b, vars);
        std::vector<int64_t> int_coeffs;
        for (size_t i = 0; i < coeffs_a.size(); i++) {
          int_coeffs.push_back(*as_const_int(coeffs_a[i]) - *as_const_int(coeffs_b[i]));
        }
        subset.addEquality(int_coeffs);
      } else {
        LOG(FATAL) << "Unsupported constraint expression: " << entry->GetTypeKey();
      }
    }
    set.unionInPlace(subset);
  }
}

IntegerSet::IntegerSet(const PrimExpr& constraint, const Array<Var>& vars) {
  auto constraints_union = ExtractComponents(constraint);
  auto space = PresburgerSpace::getSetSpace(vars.size(), 0, 0);
  PresburgerSet presburger_set = PresburgerSet::getEmpty(space);
  auto node = make_object<IntegerSetNode>(presburger_set, vars);
  Update(constraint, vars, node->set);
  node->SetVars(vars);
  data_ = std::move(node);
}

IntegerSet::IntegerSet(const PresburgerSet &set, const Array<Var>& vars) {
  auto node = make_object<IntegerSetNode>(set);
  node->SetVars(vars);
  data_ = std::move(node);
}

void IntegerSetNode::UpdateConstraint(const PrimExpr& constraint, const Array<Var>& vars) {
  Update(constraint, vars, set);
  SetVars(vars);
}

PrimExpr IntegerSetNode::GenerateConstraint() const {
  PrimExpr constraint = Bool(0);
  for (const IntegerRelation &disjunct : set.getAllDisjuncts()) {
    PrimExpr union_entry = Bool(1);
    for (unsigned i = 0, e = disjunct.getNumEqualities(); i < e; ++i) {
      PrimExpr linear_eq = IntImm(DataType::Int(32), 0);
      if (disjunct.getNumCols() > 1) {
        for (unsigned j = 0, f = disjunct.getNumCols() - 1; j < f; ++j) {
          auto coeff = disjunct.atEq(i, j);
          if (coeff >= 0 || is_zero(linear_eq)) {
            linear_eq = linear_eq + IntImm(DataType::Int(32), coeff) * vars[j];
          } else {
            linear_eq = linear_eq - IntImm(DataType::Int(32), -coeff) * vars[j];
          }
        }
      }
      auto c0 = disjunct.atEq(i, disjunct.getNumCols() - 1);
      linear_eq = linear_eq + IntImm(DataType::Int(32), c0);
      union_entry = (union_entry && (linear_eq == 0));
    }
    for (unsigned i = 0, e = disjunct.getNumInequalities(); i < e; ++i) {
      PrimExpr linear_eq = IntImm(DataType::Int(32), 0);
      if (disjunct.getNumCols() > 1) {
        for (unsigned j = 0, f = disjunct.getNumCols() - 1; j < f; ++j) {
          auto coeff = disjunct.atIneq(i, j);
          if (coeff >= 0 || is_zero(linear_eq)) {
            linear_eq = linear_eq + IntImm(DataType::Int(32), coeff) * vars[j];
          } else {
            linear_eq = linear_eq - IntImm(DataType::Int(32), -coeff) * vars[j];
          }
        }
      }
      auto c0 = disjunct.atIneq(i, disjunct.getNumCols() - 1);
      if (c0 >= 0) {
        linear_eq = linear_eq + IntImm(DataType::Int(32), c0);
      } else {
        linear_eq = linear_eq - IntImm(DataType::Int(32), -c0);
      }
      union_entry = (union_entry && (linear_eq >= 0));
    }
    constraint = constraint || union_entry;
  }

  return constraint;
}

IntegerSet Union(const Array<IntegerSet>& sets) {
  CHECK_GT(sets.size(), 0);
  if (sets.size() == 1) return sets[0];
  auto new_set = sets[0]->set;
  for (size_t i = 1; i < sets.size(); ++i) {
    new_set.unionInPlace(sets[i]->set);
  }
  return IntegerSet(new_set, sets[0]->GetVars());
}

IntegerSet Intersect(const Array<IntegerSet>& sets) {
  CHECK_GT(sets.size(), 0);
  if (sets.size() == 1) return sets[0];
  auto new_set = sets[0]->set;
  for (size_t i = 1; i < sets.size(); ++i) {
    new_set = new_set.intersect(sets[i]->set);
  }
  return IntegerSet(new_set, sets[0]->GetVars());
}

IntSet EvalSet(const PrimExpr& e, const IntegerSet& set) {
  auto tvm_coeffs = DetectLinearEquation(e, set->GetVars());
  SmallVector<int64_t> coeffs;
  coeffs.reserve(tvm_coeffs.size());
  for (auto &it : tvm_coeffs) {
    coeffs.push_back(*as_const_int(it));
  }

  IntSet result = IntSet().Nothing();
  for (auto &it : set->set.getAllDisjuncts()) {
    Simplex simplex(it);
    auto range = simplex.computeIntegerBounds(coeffs);
    auto maxRoundedDown(
      simplex.computeOptimum(Simplex::Direction::Up, coeffs));
    auto opt = range.first.getOptimumIfBounded();
    auto min = opt.hasValue() ? IntImm(DataType::Int(64), opt.getValue()) : neg_inf();
    opt = range.second.getOptimumIfBounded();
    auto max = opt.hasValue() ? IntImm(DataType::Int(64), opt.getValue()) : pos_inf();
    auto interval = IntervalSet(min, max);
    result = Union({result, interval});
  }
  return result;
}

IntegerSet MakeIntegerSet(const PrimExpr& constraint, const Array<Var>& vars) {
  return IntegerSet(constraint, vars);
}

inline IntegerSet MakeUnion(const Array<IntegerSet>& sets) {
  return Union(sets);
}

inline IntegerSet MakeIntersect(const Array<IntegerSet>& sets) {
  return Intersect(sets);
}

TVM_REGISTER_GLOBAL("arith.IntegerSet").set_body_typed(MakeIntegerSet);

TVM_REGISTER_GLOBAL("arith.Union").set_body_typed(MakeUnion);
TVM_REGISTER_GLOBAL("arith.Intersect").set_body_typed(MakeIntersect);
TVM_REGISTER_GLOBAL("arith.EvalRange").set_body_typed(
  [](const PrimExpr& e, const IntegerSet& set) { return EvalSet(e, set); }
);

TVM_REGISTER_NODE_TYPE(IntegerSetNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IntegerSetNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto set = node.as<IntegerSetNode>();
      ICHECK(ret) << "Unknown type:" << node->GetTypeKey();
      p->stream << "{";
      p->stream << set->GetVars() << ": ";
      p->stream << node.as<IntegerSetNode>()->GenerateConstraint();
      p->stream << "}";
    });

}  // namespace arith
}  // namespace tvm